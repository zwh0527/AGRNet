import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        # e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class SelfTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # feed-forward network
        self.FF = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_mask=None):
        """
        Args:
            x (torch.Tensor): [B, C, N, 1]
            x_mask (torch.Tensor): [B, N, 1](optional)
        return:
            y (torch.Tensor): [B, C, N, 1]
        """
        bs, h = x.size(0), x.size(2)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')

        query, key, value = x, x, x

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=x_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.FF(torch.cat([x, message], dim=2))
        message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x + message


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.short_cut = None
        if out_channels != channels:
            self.short_cut = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.short_cut:
            out = out + self.short_cut(x)
        else:
            out = out + x
        return out


class AGRBlock(nn.Module):
    def __init__(self, in_channels, net_channels, embed_nums,
                 point_transformer_nums, corr_transformer_nums, nhead,
                 init=True):
        super().__init__()

        self.init = init
        
        # 1 point embedding
        point_embed = [PointCN(net_channels) for i in range(embed_nums - 1)]
        point_embed.insert(0, PointCN(in_channels, net_channels))
        self.point_embed = nn.Sequential(*point_embed)

        if init:
            # 2. point transformer
            point_transformer = [SelfTransformerBlock(net_channels, nhead) for i in
                                 range(point_transformer_nums)]
            self.point_transformer = nn.Sequential(*point_transformer)

            # 3. merge
            self.merge = PointCN(net_channels * 2, net_channels)

        # 4. correspondence transformer
        corr_transformer = [SelfTransformerBlock(net_channels, nhead) for i in
                            range(corr_transformer_nums)]
        self.corr_transformer = nn.Sequential(*corr_transformer)

        # 5. predictor
        self.predictor = nn.Sequential(nn.Conv2d(net_channels, net_channels, 1, bias=False),
                                       nn.InstanceNorm2d(net_channels, eps=1e-3),
                                       nn.BatchNorm2d(net_channels),
                                       nn.ReLU(True),
                                       nn.Conv2d(net_channels, 1, 1, bias=False))

    def forward(self, input, xs):
        """
        Args:
            input (torch.Tensor): [B, 4/6/8, N, 1]
            when the first block and without side information: 4
            when the first block and with side information: 6
            when not the first block and without side information: 6
            when not the first block and with side information: 8

            xs (torch.Tensor): [B, 1, N, 4]
        return:
            logits (torch.Tensor): [B, N]
            e_hat (torch.Tensor): [B, 9]
            residuals (torch.Tensor): [B, 1, N, 1]
        """
        # 1. point embedding
        if self.init:
            x1 = self.point_embed(torch.cat([input[:, :2, :, :], input[:, 4:, :, :]], dim=1))
            x2 = self.point_embed(input[:, 2:, :, :])

            # 2. point transformer
            x1 = self.point_transformer(x1)
            x2 = self.point_transformer(x2)
            corr = self.merge(torch.cat([x1, x2], dim=1))
        else:
            corr = self.point_embed(input)

        # 3. correspondence transformer
        corr = self.corr_transformer(corr)

        # 4. predictor
        logits = self.predictor(corr).squeeze(-1).squeeze(1)  # [B, 1, N, 1] -> [B, N]
        e_hat = weighted_8points(xs, logits)  # [B, 9]
        residual = batch_episym(xs[:, 0, :, :2], xs[:, 0, :, 2:], e_hat).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        return logits, e_hat, residual, corr


class FinalPredictor(nn.Module):
    def __init__(self, channel, iter_nums=2, nhead=4):
        super().__init__()
        self.iter_nums = iter_nums
        self.self_attention = nn.Sequential(PointCN(channel * iter_nums, channel),
                                            SelfTransformerBlock(channel, nhead))
        self.predictor = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
                                       nn.InstanceNorm2d(channel, eps=1e-3),
                                       nn.BatchNorm2d(channel),
                                       nn.ReLU(True),
                                       nn.Conv2d(channel, 1, 1, bias=False))

    def forward(self, corr_cat):
        return self.predictor(self.self_attention(corr_cat))


class AGRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_nums = config.iter_nums
        self.side_channel = (config.use_ratio == 2) + (config.use_mutual == 2)
        self.init_block = AGRBlock(2 + self.side_channel, config.net_channels,
                                   config.embed_nums, config.point_transformer_nums, config.corr_transformer_nums,
                                   config.transformer_heads, True)
        self.iter_block = nn.ModuleList([AGRBlock(4 + self.side_channel, config.net_channels,
                                                  config.embed_nums, config.point_transformer_nums,
                                                  config.corr_transformer_nums,
                                                  config.transformer_heads, True) for _ in
                                         range(config.iter_nums - 1)])
        self.is_final_predictor = config.is_final_predictor
        if self.is_final_predictor:
            self.final_predictor = FinalPredictor(config.net_channels, config.iter_nums, config.transformer_heads)

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1, 2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat, res_corr = [], [], []
        logits, e_hat, residual, corr = self.init_block(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat), res_corr.append(corr)
        for i in range(self.iter_nums - 1):
            logits, e_hat, residual, corr = self.iter_block[i](
                torch.cat([input, residual.detach(),
                           torch.relu(torch.tanh(logits)).reshape(
                               residual.shape).detach()],
                          dim=1), data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat), res_corr.append(corr)
        if self.is_final_predictor:
            logits = self.final_predictor(torch.cat(res_corr, dim=1)).squeeze(-1).squeeze(1)
            e_hat = weighted_8points(data['xs'], logits)
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat
