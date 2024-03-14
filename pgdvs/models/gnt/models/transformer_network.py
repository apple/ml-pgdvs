import numpy as np

import torch
import torch.nn as nn

from pgdvs.models.gnt.common import TINY_NUMBER, HUGE_NUMBER


# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        # - q: [#ray, #sample, feat_dim];
        # - k: [#ray, #sample, #src, feat_dim];
        # - pos: [#ray, #sample, #src, 4];
        # - mask: [#ray, #sample, #src, 1]
        q = self.q_fc(q)  # [#ray, #sample, feat_dim]
        k = self.k_fc(k)  # [#ray, #sample, #src, feat_dim]
        v = self.v_fc(k)  # [#ray, #sample, #src, feat_dim]

        n_ray, n_sample, n_src, feat_dim = k.shape
        mask_invalid_cnt = torch.sum((mask == 0).float(), dim=2)  # [#ray, #sample, 1]
        unique_invalid_cnt = torch.unique(mask_invalid_cnt).long().tolist()

        flat_k = k.reshape((n_ray * n_sample, n_src, feat_dim))
        flat_invalid_cnt = mask_invalid_cnt.reshape((n_ray * n_sample))
        flat_valid_mask = (mask != 0).reshape((n_ray * n_sample, n_src))
        flat_idxs = torch.arange(flat_k.shape[0]).to(k.device)  # [#ray x #sample]

        shuffled_flat_idxs = []

        k_std = []
        normalized_k_std = []

        for invalid_cnt in unique_invalid_cnt:
            unique_mask = flat_invalid_cnt == invalid_cnt

            valid_cnt = n_src - invalid_cnt

            n_unique = torch.sum(unique_mask)
            unique_flat_k = flat_k[unique_mask, :, :]  # [#unique, #src, feat_dim]

            shuffled_flat_idxs.append(flat_idxs[unique_mask])

            if valid_cnt in [1]:
                unique_k_std = torch.zeros_like(unique_flat_k[:, 0, :])
                unique_normalized_k_std = torch.zeros_like(unique_k_std)
            else:
                if valid_cnt not in [0, n_src]:
                    unique_valid_mask = flat_valid_mask[
                        unique_mask, :
                    ]  # [#unique, #src]

                    unique_flat_k = unique_flat_k[unique_valid_mask, :].reshape(
                        (n_unique, valid_cnt, feat_dim)
                    )

                if valid_cnt == 0:
                    # remove mask
                    flat_valid_mask[unique_mask, :] = True
                    mask = flat_valid_mask.reshape((n_ray, n_sample, n_src, 1)).float()
                    flat_invalid_cnt[unique_mask] = 0
                    mask_invalid_cnt = flat_invalid_cnt.reshape((n_ray, n_sample, 1))

                unique_k_std = torch.std(unique_flat_k, dim=1)  # [#unique, feat_dim]
                unique_normalized_k_std = unique_k_std / (
                    torch.mean(torch.abs(unique_flat_k), dim=1) + TINY_NUMBER
                )

            k_std.append(unique_k_std)
            normalized_k_std.append(unique_normalized_k_std)

        sorted_flat_idxs = torch.argsort(torch.cat(shuffled_flat_idxs, dim=0))
        k_std = torch.cat(k_std, dim=0)[sorted_flat_idxs, ...].reshape(
            (n_ray, n_sample, feat_dim)
        )
        normalized_k_std = torch.cat(normalized_k_std, dim=0)[
            sorted_flat_idxs, ...
        ].reshape((n_ray, n_sample, feat_dim))

        assert torch.all(~torch.isnan(k_std)), f"{torch.sum(torch.isnan(k_std))}"
        assert torch.all(
            ~torch.isnan(normalized_k_std)
        ), f"{torch.sum(torch.isnan(normalized_k_std))}"

        pos = self.pos_fc(pos)  # [#ray, #sample, #src, feat_dim]
        attn = k - q[:, :, None, :] + pos  # [#ray, #sample, #src, feat_dim]
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -float("inf"))
        attn = torch.softmax(attn, dim=-2)  # [#ray, #sample, #src, feat_dim]

        # NOTE: change all inf attn to uniform
        mask_all_invalid = mask_invalid_cnt == n_src  # [#ray, #sample, 1]
        idx_ray, idx_sample, _ = torch.nonzero(mask_all_invalid, as_tuple=True)
        uniform_dist = torch.ones_like(attn[0, 0, :, :]) / n_src  # [#src, feat_dim]
        attn[idx_ray, idx_sample, :, :] = uniform_dist[None, :, :]

        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)  # [#ray, #sample, feat_dim]
        x = self.dp(self.out_fc(x))  # [#ray, #sample, feat_dim]
        return x, attn, k_std, normalized_k_std

    def archive_forward(self, q, k, pos, mask=None):
        # - q: [#ray, #sample, feat_dim];
        # - k: [#ray, #sample, #src, feat_dim];
        # - pos: [#ray, #sample, #src, 4];
        # - mask: [#ray, #sample, #src, 1]
        q = self.q_fc(q)  # [#ray, #sample, feat_dim]
        k = self.k_fc(k)  # [#ray, #sample, #src, feat_dim]
        v = self.v_fc(k)  # [#ray, #sample, #src, feat_dim]

        k_std = torch.std(k, dim=2)  # [#ray, #sample, feat_dim]
        normalized_k_std = k_std / (torch.mean(torch.abs(k), dim=2) + TINY_NUMBER)

        pos = self.pos_fc(pos)  # [#ray, #sample, #src, feat_dim]
        attn = k - q[:, :, None, :] + pos  # [#ray, #sample, #src, feat_dim]
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)  # [#ray, #sample, #src, feat_dim]
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)  # [#ray, #sample, feat_dim]
        x = self.dp(self.out_fc(x))  # [#ray, #sample, feat_dim]
        return x, attn, k_std, normalized_k_std


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        # - q: [#ray, #sample, feat_dim];
        # - k: [#ray, #sample, #src, feat_dim];
        # - pos: [#ray, #sample, #src, 4];
        # - mask: [#ray, #sample, #src, 1]
        residue = q
        x = self.attn_norm(q)  # [#ray, #sample, feat_dim]
        x, attn, k_std, normalized_k_std = self.attn(
            x, k, pos, mask
        )  # x: [#ray, #sample, feat_dim]; attn: [#ray, #sample, #src, feat_dim]
        x = x + residue

        residue = x
        x = self.ff_norm(x)  # [#ray, #sample, feat_dim]
        x = self.ff(x)  # [#ray, #sample, feat_dim]
        x = x + residue

        return x, attn, k_std, normalized_k_std


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)  # [#ray, #sample, feat_dim]
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(
                0, 2, 1, 3
            )  # [#ray, #sample, #head, feat_dim] -> [#ray, #head, #sample, feat_dim]
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(
                0, 2, 1, 3
            )  # [#ray, #sample, #head, feat_dim] -> [#ray, #head, #sample, feat_dim]
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(
            0, 2, 1, 3
        )  # [#ray, #sample, #head, feat_dim] -> [#ray, #head, #sample, feat_dim]

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
                q.shape[-1]
            )  # [#ray, #head, #sample, #sample]
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(
                0, 3, 1, 2
            )
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(
                0, 3, 1, 2
            )
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = (
            torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        )  # [#ray, #head, #sample, #sample] x [#ray, #head, #sample, feat_dim] -> [#ray, #head, #sample, feat_dim] -> [#ray, #sample, #head, feat_dim]
        out = out.view(
            x.shape[0], x.shape[1], -1
        )  # [#ray, #sample, #head, feat_dim] -> [#ray, #sample, #head x feat_dim]
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        ff_hid_dim,
        ff_dp_rate,
        n_heads,
        attn_dp_rate,
        attn_mode="qk",
        pos_dim=None,
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim)

    def forward(self, x, pos=None, ret_attn=False):
        residue = x  # [#ray, #sample, feat_dim]
        x = self.attn_norm(x)  # [#ray, #sample, feat_dim]
        x = self.attn(x, pos, ret_attn)
        if ret_attn:
            (
                x,
                attn,
            ) = x  # x: [#ray, #sample, feat_dim]; attn: [#rays, #head, #sample, #sample]
        x = x + residue

        residue = x  # [#ray, #sample, feat_dim]
        x = self.ff_norm(x)  # [#ray, #sample, feat_dim]
        x = self.ff(x)  # [#ray, #sample, feat_dim]
        x = x + residue  # [#ray, #sample, feat_dim]

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x


class GNT(nn.Module):
    def __init__(
        self,
        *,
        netwidth,
        transformer_depth,
        in_feat_ch=32,
        posenc_max_freq_log2=9,
        pos_enc_n_freqs=10,
        view_enc_n_freqs=10,
        ret_alpha=True,
    ):
        super(GNT, self).__init__()

        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=posenc_max_freq_log2,
            num_freqs=pos_enc_n_freqs,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=posenc_max_freq_log2,
            num_freqs=view_enc_n_freqs,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.posenc_dim = self.pos_enc.out_dim
        self.viewenc_dim = self.view_enc.out_dim

        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(netwidth)
        self.rgb_fc = nn.Linear(netwidth, 3)
        self.relu = nn.ReLU()

        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, netwidth),
            nn.ReLU(),
            nn.Linear(netwidth, netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])

        for i in range(transformer_depth):
            # view transformer
            view_trans = Transformer2D(
                dim=netwidth,
                ff_hid_dim=int(netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)

            # ray transformer
            ray_trans = Transformer(
                dim=netwidth,
                ff_hid_dim=int(netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)

            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(netwidth + self.posenc_dim + self.viewenc_dim, netwidth),
                    nn.ReLU(),
                    nn.Linear(netwidth, netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

    def forward(
        self,
        rgb_feat,
        ray_diff,
        mask,
        pts,
        ray_d,
        ret_view_entropy=False,
        ret_view_std=False,
    ):
        # - rgb_feat: [#ray, #sample, #src, 3 + feat_d]
        # - ray_diff: [#ray, #sample, #src, 4]
        # - mask: [#ray, #sample, #src, 1]
        # - pts: [#ray, #sample, 3] or [#ray, #sample, #src, 3]
        # - ray_d: [#ray, 3]

        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # [#ray, 3]
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)  # [#ray, pos_enc_dim]
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)  # [#ray x #sample, pos_enc_dim]
        pts_ = torch.reshape(
            pts_, list(pts.shape[:-1]) + [pts_.shape[-1]]
        )  # [#ray, #sample, pos_enc_dim]
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)  # [#ray, #sample, pos_enc_dim]
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(
            embed, [self.posenc_dim, self.viewenc_dim], dim=-1
        )  # both of [#ray, #sample, pos_enc_dim]

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)  # [#ray, #sample, #src, netwidth]
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]  # [#ray, #sample, netwidth]

        if ret_view_entropy:
            # NOTE: DEBUG. Compute entropy across views
            view_entropy_list = []

        if ret_view_std:
            view_std_list = [
                torch.mean(torch.std(rgb_feat, dim=2), dim=2)
            ]  # [#ray, #sample]
            tmp_view_std_normalized = torch.std(rgb_feat, dim=2) / (
                torch.mean(torch.abs(rgb_feat), dim=2) + TINY_NUMBER
            )
            view_std_normalized_list = [
                torch.mean(tmp_view_std_normalized, dim=2)
            ]  # [#ray, #sample]

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q, view_attn, view_std, view_std_normalized = crosstrans(
                q, rgb_feat, ray_diff, mask
            )  # q: [#ray, #sample, netwidth]; view_attn: [#ray, #sample, #src, feat_dim]; view_std/view_std_normalized: [#ray, #sample, feat_dim]

            # embed positional information
            if i % 2 == 0:
                q = torch.cat(
                    (q, input_pts, input_views), dim=-1
                )  # [#ray, #sample, sum_feat_dim]
                q = q_fc(q)  # [#ray, #sample, netwidth]
            # ray transformer
            q = selftrans(q, ret_attn=self.ret_alpha)  # [#ray, #sample, netwidth]
            # 'learned' density
            if self.ret_alpha:
                q, attn = q  # attn: [#rays, #sample]

            if ret_view_entropy:
                # NOTE: DEBUG. Compute entropy across views
                # print("\nview_attn: ", torch.mean(torch.sum(view_attn, dim=2)), torch.std(torch.sum(view_attn, dim=2)), "\n")
                tmp_view_attn_sum = torch.sum(view_attn, dim=2)
                assert torch.all(
                    ~torch.isnan(tmp_view_attn_sum)
                ), f"{torch.sum(torch.isnan(tmp_view_attn_sum))}"
                assert torch.all(
                    torch.abs(tmp_view_attn_sum - 1) < 1e-5
                ), f"{torch.min(tmp_view_attn_sum)}, {torch.max(tmp_view_attn_sum)}"
                tmp_entropy = torch.sum(
                    -1 * view_attn * torch.log(view_attn + 1e-8), dim=2
                )  # [#ray, #sample, feat_dim]
                tmp_entropy = torch.mean(tmp_entropy, dim=2)  # [#ray, #sample]
                view_entropy_list.append(tmp_entropy)

            if ret_view_std:
                view_std_list.append(torch.mean(view_std, dim=2))
                view_std_normalized_list.append(torch.mean(view_std_normalized, dim=2))

        extra_infos = {}
        if ret_view_entropy:
            view_entropy = torch.stack(
                view_entropy_list, dim=2
            )  # [#ray, #sample, #layer]
            extra_infos["view_entropy"] = view_entropy

        if ret_view_std:
            view_std = torch.stack(
                view_std_list, dim=2
            )  # [#ray, #sample, #layer + 1], 1 for the raw RGB feat
            view_std_normalized = torch.stack(
                view_std_normalized_list, dim=2
            )  # [#ray, #sample, #layer + 1], 1 for the raw RGB feat
            extra_infos["view_std"] = view_std
            extra_infos["view_std_normalized"] = view_std_normalized

        # normalize & rgb
        h = self.norm(q)  # [#ray, #sample, netwidth]
        outputs = self.rgb_fc(h.mean(dim=1))  # [#ray, netwidth] -> [#ray, 3]
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1), extra_infos
        else:
            return outputs, extra_infos
