from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError("patch_size must be an int or length-2 sequence")
    return (int(patch_size[0]), int(patch_size[1]))


def split_patch_angles(image: torch.Tensor, patch_size: int | Sequence[int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W]")
    patch_h, patch_w = _normalize_patch_size(patch_size)
    unfold = F.unfold(image.unsqueeze(0), kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches = unfold.squeeze(0).t()
    return patches.clamp(0, 1).float() * 2.0 * math.pi


def measurement_dim(num_qubits: int) -> int:
    return num_qubits * 2


def _wire_index(qubit: int, num_qubits: int) -> int:
    return num_qubits - qubit - 1


def _reorder_features(features: torch.Tensor, num_qubits: int) -> torch.Tensor:
    single = torch.flip(features[..., :num_qubits], dims=[-1])
    zz = features[..., num_qubits:]
    zz_order = list(range(num_qubits - 2, -1, -1)) + [num_qubits - 1]
    zz = zz[..., zz_order]
    return torch.cat([single, zz], dim=-1)


def _encode_params(angles: torch.Tensor, num_qubits: int, encoding: str) -> None:
    idx = 0
    total = len(angles)
    encoding = encoding.lower()
    if encoding in {"ryrx", "ry_rx"}:
        while idx < total:
            for q in range(num_qubits):
                if idx >= total:
                    break
                qml.RY(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
            for q in range(num_qubits):
                if idx >= total:
                    break
                qml.RX(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
        return
    if encoding in {"rxry", "rx_ry"}:
        while idx < total:
            any_applied = False
            for q in range(num_qubits):
                if idx >= total:
                    break
                qml.RX(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
                any_applied = True
                if idx >= total:
                    break
                qml.RY(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
            if not any_applied:
                break
            for q in range(num_qubits):
                qml.CNOT(
                    wires=[
                        _wire_index(q, num_qubits),
                        _wire_index((q + 1) % num_qubits, num_qubits),
                    ]
                )
        return
    if encoding in {"rxryrz", "rx_ry_rz", "xyz"}:
        while idx < total:
            for q in range(num_qubits):
                if idx >= total:
                    break
                qml.RX(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
                if idx >= total:
                    break
                qml.RY(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
                if idx >= total:
                    break
                qml.RZ(angles[idx], wires=_wire_index(q, num_qubits))
                idx += 1
        return
    raise ValueError("encoding must be 'ryrx', 'rxry', or 'rxryrz'")


def _add_vqc_layers(theta: torch.Tensor, start: int, num_layers: int, num_qubits: int) -> int:
    idx = start
    for _ in range(num_layers):
        for q in range(num_qubits):
            qml.CNOT(
                wires=[
                    _wire_index(q, num_qubits),
                    _wire_index((q + 1) % num_qubits, num_qubits),
                ]
            )
        for q in range(num_qubits):
            qml.RX(theta[idx], wires=_wire_index(q, num_qubits))
            idx += 1
        for q in range(num_qubits):
            qml.RY(theta[idx], wires=_wire_index(q, num_qubits))
            idx += 1
    return idx


@dataclass
class QuantumAnsatz:
    data_dim: int
    num_qubits: int = 8
    vqc_layers: int = 1
    reuploading: int = 1
    encoding: str | None = None
    backend_device: str = "cpu"
    use_torch_autograd: bool = False

    def __post_init__(self) -> None:
        if self.encoding is None:
            self.encoding = "rxry"
        self.encoding = self.encoding.lower()
        if self.reuploading < 1:
            raise ValueError("reuploading must be >= 1")
        self._feature_device = qml.device("default.qubit", wires=self.num_qubits)
        self._state_device = qml.device("default.qubit", wires=self.num_qubits)
        self._feature_qnode = qml.QNode(
            self._feature_circuit,
            self._feature_device,
            interface="torch",
            diff_method="backprop",
        )
        self._state_qnode = qml.QNode(
            self._state_circuit,
            self._state_device,
            interface="torch",
            diff_method="backprop",
        )

    @property
    def param_shape(self) -> int:
        return self.reuploading * self.vqc_layers * 2 * self.num_qubits

    @property
    def feature_dim(self) -> int:
        return measurement_dim(self.num_qubits)

    def _apply_circuit(self, angles: torch.Tensor, theta: torch.Tensor) -> None:
        theta_idx = 0
        for _ in range(self.reuploading):
            _encode_params(angles, self.num_qubits, self.encoding)
            theta_idx = _add_vqc_layers(theta, theta_idx, self.vqc_layers, self.num_qubits)

    def _feature_circuit(self, angles: torch.Tensor, theta: torch.Tensor):
        self._apply_circuit(angles, theta)
        measurements = [qml.expval(qml.PauliZ(_wire_index(q, self.num_qubits))) for q in range(self.num_qubits)]
        measurements.extend(
            qml.expval(
                qml.PauliZ(_wire_index(q, self.num_qubits))
                @ qml.PauliZ(_wire_index((q + 1) % self.num_qubits, self.num_qubits))
            )
            for q in range(self.num_qubits)
        )
        return tuple(measurements)

    def _state_circuit(self, angles: torch.Tensor, theta: torch.Tensor):
        self._apply_circuit(angles, theta)
        return qml.state()

    def circuit_for_angles(self, angles: Sequence[float], param_values: Sequence[float] | None = None):
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        if len(angles) != self.data_dim:
            raise ValueError(f"angles length {len(angles)} does not match data_dim {self.data_dim}")
        angles_t = torch.as_tensor(angles, dtype=torch.float32)
        theta_t = torch.as_tensor(param_values, dtype=torch.float32)
        return qml.draw(self._feature_qnode)(angles_t, theta_t)

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        angles_t = torch.as_tensor(patch_angles, dtype=torch.float32)
        theta_t = torch.as_tensor(
            param_values if param_values is not None else [0.0] * self.param_shape,
            dtype=torch.float32,
        )
        return self.torch_features(angles_t, theta_t).detach().cpu().numpy()

    def torch_features(
        self, patch_angles: torch.Tensor, theta: torch.Tensor, return_statevector: bool = False
    ) -> torch.Tensor:
        if theta.dim() != 1:
            raise ValueError("theta must be 1-D")
        if patch_angles.dim() == 1:
            patch_angles = patch_angles.unsqueeze(0)
        if patch_angles.dim() != 2:
            raise ValueError("patch_angles must have shape [B, D] or [D]")
        if patch_angles.shape[-1] != self.data_dim:
            raise ValueError(f"patch_angles last dim {patch_angles.shape[-1]} does not match data_dim {self.data_dim}")

        target_device = theta.device
        theta_qnode = theta.to(device="cpu", dtype=theta.dtype)
        patch_angles = patch_angles.to(device="cpu", dtype=theta.dtype)
        features = []
        states = [] if return_statevector else None

        for row in patch_angles:
            vals = self._feature_qnode(row, theta_qnode)
            feats = torch.stack(list(vals)).to(device=target_device, dtype=theta.dtype)
            feats = _reorder_features(feats, self.num_qubits)
            features.append(feats)
            if return_statevector:
                state = self._state_qnode(row, theta_qnode).to(device=target_device)
                states.append(state)

        out = torch.stack(features, dim=0)
        if return_statevector:
            state_out = torch.stack(states, dim=0)
            if out.shape[0] == 1:
                return out.squeeze(0), state_out.squeeze(0)
            return out, state_out
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out


@dataclass
class QuantumPatchModel:
    patch_size: int | Sequence[int] = 4
    channels: int = 2
    num_qubits: int = 8
    vqc_layers: int = 1
    reuploading: int = 1
    encoding: str | None = None
    backend_device: str = "cpu"
    use_torch_autograd: bool = False

    def __post_init__(self) -> None:
        self.patch_size = _normalize_patch_size(self.patch_size)
        patch_h, patch_w = self.patch_size
        data_dim = self.channels * patch_h * patch_w
        self.ansatz = QuantumAnsatz(
            data_dim=data_dim,
            num_qubits=self.num_qubits,
            vqc_layers=self.vqc_layers,
            reuploading=self.reuploading,
            encoding=self.encoding,
            backend_device=self.backend_device,
            use_torch_autograd=self.use_torch_autograd,
        )

    @property
    def param_shape(self) -> int:
        return self.ansatz.param_shape

    def circuit_for_angles(self, angles: Sequence[float], param_values: Sequence[float] | None = None):
        return self.ansatz.circuit_for_angles(angles, param_values)

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        return self.ansatz.features(patch_angles, param_values)

    def image_patch_features(
        self, image: torch.Tensor, param_values: Sequence[float] | None = None
    ) -> List[np.ndarray]:
        angles = split_patch_angles(image, self.patch_size)
        return [self.features(a.tolist(), param_values) for a in angles]


@dataclass
class SeparateQKV:
    query_ansatz: QuantumAnsatz
    key_ansatz: QuantumAnsatz
    value_ansatz: QuantumAnsatz

    def qkv_from_patch(
        self,
        patch_angles: Sequence[float],
        params_q: Sequence[float] | None = None,
        params_k: Sequence[float] | None = None,
        params_v: Sequence[float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.query_ansatz.features(patch_angles, params_q)
        k = self.key_ansatz.features(patch_angles, params_k)
        v = self.value_ansatz.features(patch_angles, params_v)
        return q, k, v

    def qkv_from_image(
        self,
        image: torch.Tensor,
        patch_size: int | Sequence[int],
        params_q: Sequence[float] | None = None,
        params_k: Sequence[float] | None = None,
        params_v: Sequence[float] | None = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        angles = split_patch_angles(image, patch_size)
        return [self.qkv_from_patch(a.tolist(), params_q, params_k, params_v) for a in angles]


class SeparateQKVProjector(nn.Module):
    def __init__(
        self,
        ansatz_q: QuantumAnsatz,
        ansatz_k: QuantumAnsatz,
        ansatz_v: QuantumAnsatz,
        device: torch.device | str | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.ansatz_q = ansatz_q
        self.ansatz_k = ansatz_k
        self.ansatz_v = ansatz_v
        self.device = torch.device(device) if device is not None else None
        self.trainable = trainable
        if trainable:
            limit_q = math.sqrt(6.0 / (ansatz_q.param_shape + ansatz_q.feature_dim))
            limit_k = math.sqrt(6.0 / (ansatz_k.param_shape + ansatz_k.feature_dim))
            limit_v = math.sqrt(6.0 / (ansatz_v.param_shape + ansatz_v.feature_dim))
            self.theta_q = nn.Parameter(
                (torch.rand(ansatz_q.param_shape, dtype=torch.float32) * 2 - 1) * limit_q * math.pi
            )
            self.theta_k = nn.Parameter(
                (torch.rand(ansatz_k.param_shape, dtype=torch.float32) * 2 - 1) * limit_k * math.pi
            )
            self.theta_v = nn.Parameter(
                (torch.rand(ansatz_v.param_shape, dtype=torch.float32) * 2 - 1) * limit_v * math.pi
            )

    def forward_image(
        self, image: torch.Tensor, patch_size: int | Sequence[int], param_values=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        angles = split_patch_angles(image, patch_size)
        angle_tensor = torch.stack([a for a in angles], dim=0).to(self.device or angles.device)
        if self.trainable:
            q = self.ansatz_q.torch_features(angle_tensor, self.theta_q)
            k = self.ansatz_k.torch_features(angle_tensor, self.theta_k)
            v = self.ansatz_v.torch_features(angle_tensor, self.theta_v)
        else:
            pv = param_values or {}
            q_params = pv.get("q")
            k_params = pv.get("k")
            v_params = pv.get("v")
            q = torch.as_tensor(
                np.stack([self.ansatz_q.features(a.tolist(), q_params) for a in angles]),
                dtype=torch.float32,
                device=self.device,
            )
            k = torch.as_tensor(
                np.stack([self.ansatz_k.features(a.tolist(), k_params) for a in angles]),
                dtype=torch.float32,
                device=self.device,
            )
            v = torch.as_tensor(
                np.stack([self.ansatz_v.features(a.tolist(), v_params) for a in angles]),
                dtype=torch.float32,
                device=self.device,
            )
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if k.dim() == 1:
            k = k.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return q, k, v

    def forward_angles(
        self, angles: torch.Tensor, param_values=None, return_statevector: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if angles.dim() == 2:
            angles = angles.unsqueeze(0)
        if angles.dim() != 3:
            raise ValueError("angles must have shape [B, P, D] or [P, D]")
        if return_statevector and not self.trainable:
            raise ValueError("return_statevector requires trainable=True.")
        batch, patches, feat_dim = angles.shape
        flat = angles.reshape(batch * patches, feat_dim)

        if self.trainable:
            if return_statevector:
                q, q_sv = self.ansatz_q.torch_features(flat, self.theta_q, return_statevector=True)
                k, k_sv = self.ansatz_k.torch_features(flat, self.theta_k, return_statevector=True)
                v, v_sv = self.ansatz_v.torch_features(flat, self.theta_v, return_statevector=True)
            else:
                q = self.ansatz_q.torch_features(flat, self.theta_q)
                k = self.ansatz_k.torch_features(flat, self.theta_k)
                v = self.ansatz_v.torch_features(flat, self.theta_v)
        else:
            pv = param_values or {}
            q_params = pv.get("q")
            k_params = pv.get("k")
            v_params = pv.get("v")
            q = torch.as_tensor(
                np.stack([self.ansatz_q.features(a.tolist(), q_params) for a in flat]),
                dtype=torch.float32,
                device=self.device,
            )
            k = torch.as_tensor(
                np.stack([self.ansatz_k.features(a.tolist(), k_params) for a in flat]),
                dtype=torch.float32,
                device=self.device,
            )
            v = torch.as_tensor(
                np.stack([self.ansatz_v.features(a.tolist(), v_params) for a in flat]),
                dtype=torch.float32,
                device=self.device,
            )
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if k.dim() == 1:
            k = k.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        q = q.reshape(batch, patches, -1)
        k = k.reshape(batch, patches, -1)
        v = v.reshape(batch, patches, -1)
        if return_statevector:
            if q_sv.dim() == 1:
                q_sv = q_sv.unsqueeze(0)
            if k_sv.dim() == 1:
                k_sv = k_sv.unsqueeze(0)
            if v_sv.dim() == 1:
                v_sv = v_sv.unsqueeze(0)
            q_sv = q_sv.reshape(batch, patches, -1)
            k_sv = k_sv.reshape(batch, patches, -1)
            v_sv = v_sv.reshape(batch, patches, -1)
            return q, k, v, q_sv, k_sv, v_sv
        return q, k, v

    def forward_batch(
        self,
        images: torch.Tensor,
        patch_size: int | Sequence[int],
        param_values=None,
        return_statevector: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.trainable:
            angles = torch.stack([split_patch_angles(img, patch_size) for img in images], dim=0)
            angles = angles.to(self.device or images.device)
            if return_statevector:
                return self.forward_angles(angles, param_values, return_statevector=True)
            return self.forward_angles(angles, param_values)
        if return_statevector:
            raise ValueError("return_statevector requires trainable=True.")
        q_list, k_list, v_list = [], [], []
        for img in images:
            q, k, v = self.forward_image(img, patch_size, param_values)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
        return torch.stack(q_list, dim=0), torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)


def _clone_ansatz(base: QuantumAnsatz, data_dim: int) -> QuantumAnsatz:
    return QuantumAnsatz(
        data_dim=data_dim,
        num_qubits=base.num_qubits,
        vqc_layers=base.vqc_layers,
        reuploading=base.reuploading,
        encoding=base.encoding,
        backend_device=base.backend_device,
        use_torch_autograd=base.use_torch_autograd,
    )


class HybridQuantumClassifier(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int | Sequence[int],
        ansatz_q: QuantumAnsatz,
        ansatz_k: QuantumAnsatz,
        ansatz_v: QuantumAnsatz,
        attn_layers: int = 1,
        rbf_gamma: float = 1.0,
        device: torch.device | str = "cpu",
        save_statevector: bool = False,
        save_statevector_epoch: int = 1,
        num_classes: int = 1,
        classification_task: str = "binary",
    ) -> None:
        super().__init__()
        task = (classification_task or "").lower()
        if task not in {"binary", "multi"}:
            raise ValueError("classification_task must be 'binary' or 'multi'.")
        self.classification_task = task
        self.device = torch.device(device)
        self.patch_size = _normalize_patch_size(patch_size)
        self.patch_count = (image_size // self.patch_size[0]) * (image_size // self.patch_size[1])
        if ansatz_q.feature_dim != ansatz_k.feature_dim or ansatz_q.feature_dim != ansatz_v.feature_dim:
            raise ValueError("ansatz_q, ansatz_k, ansatz_v must have the same feature_dim")
        self.attn_dim = ansatz_v.feature_dim
        if attn_layers < 1:
            raise ValueError("attn_layers must be >= 1")
        self.attn_layers = nn.ModuleList([AttentionLayer(rbf_gamma) for _ in range(attn_layers)])
        self.qkv_layers = nn.ModuleList(
            [
                SeparateQKVProjector(
                    ansatz_q=ansatz_q,
                    ansatz_k=ansatz_k,
                    ansatz_v=ansatz_v,
                    device=self.device,
                    trainable=True,
                )
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.attn_dim) for _ in range(attn_layers - 1)])
        if attn_layers > 1:
            hidden_dim = self.attn_dim
            for _ in range(1, attn_layers):
                self.qkv_layers.append(
                    SeparateQKVProjector(
                        ansatz_q=_clone_ansatz(ansatz_q, hidden_dim),
                        ansatz_k=_clone_ansatz(ansatz_k, hidden_dim),
                        ansatz_v=_clone_ansatz(ansatz_v, hidden_dim),
                        device=self.device,
                        trainable=True,
                    )
                )
        self.classifier = ClassifierHead(in_dim=self.attn_dim * self.patch_count, out_dim=num_classes)
        self.save_statevector = save_statevector
        self.save_statevector_epoch = save_statevector_epoch
        self.current_epoch = 0
        self.save_statevector_active = False
        self.saved_statevectors: dict[int, list[dict[str, torch.Tensor]]] = {}
        self.to(self.device)

    def _angles_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.tanh(feats)

    def configure_statevector_saving(self, epoch: int, active: bool, reset_storage: bool = False) -> None:
        self.current_epoch = epoch
        should_save = (
            active
            and self.save_statevector
            and self.save_statevector_epoch > 0
            and epoch % self.save_statevector_epoch == 0
        )
        self.save_statevector_active = should_save
        if should_save and reset_storage:
            self.saved_statevectors = {}

    def _record_statevectors(self, layer_idx: int, q_sv: torch.Tensor, k_sv: torch.Tensor, v_sv: torch.Tensor) -> None:
        self.saved_statevectors.setdefault(
            layer_idx,
            [],
        ).append({"epoch": self.current_epoch, "q": q_sv.detach().cpu(), "k": k_sv.detach().cpu(), "v": v_sv.detach().cpu()})

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("images must be [B, C, H, W]")

        weights_list = []
        intermediates = [] if return_intermediates else None
        x = None

        for layer_idx, attn in enumerate(self.attn_layers):
            if layer_idx == 0:
                if self.save_statevector_active:
                    q, k, v, q_sv, k_sv, v_sv = self.qkv_layers[0].forward_batch(
                        images, self.patch_size, return_statevector=True
                    )
                    self._record_statevectors(layer_idx, q_sv, k_sv, v_sv)
                else:
                    q, k, v = self.qkv_layers[0].forward_batch(images, self.patch_size)
            else:
                angles = self._angles_from_features(x)
                if self.save_statevector_active:
                    q, k, v, q_sv, k_sv, v_sv = self.qkv_layers[layer_idx].forward_angles(
                        angles, return_statevector=True
                    )
                    self._record_statevectors(layer_idx, q_sv, k_sv, v_sv)
                else:
                    q, k, v = self.qkv_layers[layer_idx].forward_angles(angles)

            q = q.to(self.device)
            k = k.to(self.device)
            v = v.to(self.device)

            x_in = v
            out, w = attn(q, k, v, return_weights=True)
            weights_list.append(w)

            x_resid = v + out
            if layer_idx < len(self.attn_layers) - 1:
                x = self.layer_norms[layer_idx](x_resid)
                x_out = x
            else:
                x = x_resid
                x_out = x_resid

            if return_intermediates:
                intermediates.append({"input": x_in, "residual": x_resid, "output": x_out})

        attn_stats = None
        if weights_list:
            w = weights_list[0]
            with torch.no_grad():
                entropy = -(w * (w + 1e-12).log()).sum(dim=-1).mean().item()
                max_w = w.max().item()
            attn_stats = {"entropy": entropy, "max_weight": max_w}

        emb = x.flatten(start_dim=1)
        logits = self.classifier(emb)
        if return_attention and return_intermediates:
            return logits, attn_stats, weights_list, intermediates
        if return_attention:
            return logits, attn_stats, weights_list
        if return_intermediates:
            return logits, attn_stats, intermediates
        return logits if attn_stats is None else (logits, attn_stats)


class AttentionLayer(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_weights: bool = False,
        return_intermediates: bool = False,
    ):
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        q_exp = q.unsqueeze(-2)
        k_exp = k.unsqueeze(-3)
        dist2 = (q_exp - k_exp).pow(2).sum(-1)
        weights = torch.softmax(-self.gamma * dist2, dim=-1)
        out = torch.matmul(weights, v)
        if return_weights:
            return out, weights
        return out


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.fc(x)
        if self.out_dim == 1:
            return out.squeeze(-1)
        return out
