from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Pauli


def _normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError("patch_size must be an int or length-2 sequence")
    return (int(patch_size[0]), int(patch_size[1]))


def split_patch_angles(image: torch.Tensor, patch_size: int | Sequence[int]) -> torch.Tensor:
    """
    Args:
        image: Tensor [C, H, W] with values in [0, 1] (normalized).
    Returns:
        angles: Tensor [num_patches, C * patch_area] in [0, 2*pi].
    """
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W]")
    patch_h, patch_w = _normalize_patch_size(patch_size)
    unfold = F.unfold(image.unsqueeze(0), kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches = unfold.squeeze(0).t()  # [num_patches, C * patch_area]
    angles = patches.clamp(0, 1).float() * 2.0 * math.pi
    return angles


def encode_params(
    circuit: QuantumCircuit, params: Sequence, num_qubits: int = 8, encoding: str = "rxry"
) -> None:
    idx = 0
    total = len(params)
    encoding = encoding.lower()
    if encoding in {"ryrx", "ry_rx"}:
        while idx < total:
            for q in range(num_qubits):
                if idx >= total:
                    break
                circuit.ry(params[idx], q)
                idx += 1
            for q in range(num_qubits):
                if idx >= total:
                    break
                circuit.rx(params[idx], q)
                idx += 1
        return
    if encoding in {"rxry", "rx_ry"}:
        while idx < total:
            any_applied = False
            for q in range(num_qubits):
                if idx >= total:
                    break
                circuit.rx(params[idx], q)
                idx += 1
                any_applied = True
                if idx >= total:
                    break
                circuit.ry(params[idx], q)
                idx += 1
            if not any_applied:
                break
            for q in range(num_qubits):
                circuit.cx(q, (q + 1) % num_qubits)
        return
    if encoding in {"rxryrz", "rx_ry_rz", "xyz"}:
        while idx < total:
            for q in range(num_qubits):
                if idx >= total:
                    break
                circuit.rx(params[idx], q)
                idx += 1
                if idx >= total:
                    break
                circuit.ry(params[idx], q)
                idx += 1
                if idx >= total:
                    break
                circuit.rz(params[idx], q)
                idx += 1
        return
    raise ValueError("encoding must be 'ryrx', 'rxry', or 'rxryrz'")


def add_vqc_layers(
    circuit: QuantumCircuit, params: ParameterVector, start: int, num_layers: int, num_qubits: int = 8
) -> int:
    """
    Adds layers and returns next free parameter index.
    """
    p = start
    for _ in range(num_layers):
        for q in range(num_qubits):
            circuit.cx(q, (q + 1) % num_qubits)
        for q in range(num_qubits):
            circuit.rx(params[p], q)
            p += 1
        for q in range(num_qubits):
            circuit.ry(params[p], q)
            p += 1
    return p


def _z_pauli(label_qubits: Iterable[int], num_qubits: int) -> Pauli:
    from qiskit.quantum_info import Pauli

    z = ["I"] * num_qubits
    for q in label_qubits:
        z[num_qubits - q - 1] = "Z"
    return Pauli("".join(z))


def _get_statevector(circuit, param_bind: dict, backend=None):
    from qiskit.quantum_info import Statevector

    if backend is None:
        bound = circuit.assign_parameters(param_bind, inplace=False)
        return Statevector.from_instruction(bound)
    job = backend.run(circuit, parameter_binds=[param_bind])
    result = job.result()
    data = np.array(result.get_statevector(circuit, param_bind), dtype=complex)
    return Statevector(data)


def z_zz_ring_features(circuit, param_bind: dict, backend=None) -> np.ndarray:
    """
    Z single-qubit expectations plus ring ZZ correlations.
    Order: [Z0..Z{n-1}, ZZ01, ZZ12, ..., ZZ{n-1,0}]
    """
    sv = _get_statevector(circuit, param_bind, backend)
    n = circuit.num_qubits
    values: List[float] = []

    for i in range(n):
        values.append(float(np.real(sv.expectation_value(_z_pauli([i], n)))))
    for i in range(n):
        j = (i + 1) % n
        values.append(float(np.real(sv.expectation_value(_z_pauli([i, j], n)))))
    return np.array(values, dtype=np.float64)


def _apply_single_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    """
    state: [B, 2**n]
    gate: [2, 2] (shared) or [B, 2, 2] (per batch)
    """
    bsz = state.shape[0]
    state = state.reshape(bsz, *([2] * num_qubits))
    state = state.movedim(qubit + 1, -1)
    state = state.reshape(bsz, -1, 2)

    if gate.dim() == 2:
        gate = gate.unsqueeze(0)
    if gate.dim() == 3 and gate.shape[0] != bsz and gate.shape[-1] == bsz:
        gate = gate.permute(2, 0, 1)
    if gate.dim() == 3 and gate.shape[0] == 1:
        gate = gate.expand(bsz, -1, -1)

    # state [B, N, 2], gate [B, 2, 2]
    state = torch.matmul(state, gate.transpose(1, 2))

    state = state.reshape(bsz, *([2] * (num_qubits - 1)), 2)
    state = state.movedim(-1, qubit + 1)
    return state.reshape(bsz, -1).contiguous()


def _apply_cnot(state: torch.Tensor, control: int, target: int, num_qubits: int) -> torch.Tensor:
    if control == target:
        return state
    bsz = state.shape[0]
    state = state.reshape(bsz, *([2] * num_qubits))
    state = state.movedim([control + 1, target + 1], [-2, -1])
    orig_shape = state.shape
    state = state.reshape(-1, 4)
    cnot = state.new_tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=state.dtype
    )
    state = torch.matmul(state, cnot.t())
    state = state.reshape(orig_shape)
    state = state.movedim([-2, -1], [control + 1, target + 1])
    return state.reshape(bsz, -1).contiguous()


def simulate_torch_features(
    angles: torch.Tensor,
    theta: torch.Tensor,
    num_qubits: int,
    vqc_layers: int,
    reuploading: int,
    encoding: str = "rxry",
    return_statevector: bool = False,
) -> torch.Tensor:
    """
    Differentiable torch simulation (statevector) with batch support.
    angles: [B, data_dim] or [data_dim]
    theta: [param_shape]
    returns: [B, feature_dim] or [feature_dim]
    """
    if reuploading < 1:
        raise ValueError("reuploading must be >= 1")
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    if theta.dim() != 1:
        raise ValueError("theta must be 1-D")
    batch = angles.shape[0]
    encoding = encoding.lower()

    device = theta.device
    dtype = torch.complex64 if theta.dtype == torch.float32 else torch.complex128
    angles = angles.to(device=device, dtype=theta.dtype)

    state = torch.zeros(batch, 2**num_qubits, device=device, dtype=dtype)
    state[:, 0] = 1.0 + 0j

    def rx_gate(phi):
        if phi.dim() == 0:
            phi = phi.expand(batch)
        c = torch.cos(phi / 2)
        s = torch.sin(phi / 2)
        gate = torch.zeros(phi.shape[0], 2, 2, device=device, dtype=dtype)
        gate[:, 0, 0] = c
        gate[:, 0, 1] = -1j * s
        gate[:, 1, 0] = -1j * s
        gate[:, 1, 1] = c
        return gate

    def ry_gate(phi):
        if phi.dim() == 0:
            phi = phi.expand(batch)
        c = torch.cos(phi / 2)
        s = torch.sin(phi / 2)
        gate = torch.zeros(phi.shape[0], 2, 2, device=device, dtype=dtype)
        gate[:, 0, 0] = c
        gate[:, 0, 1] = -s
        gate[:, 1, 0] = s
        gate[:, 1, 1] = c
        return gate

    def rz_gate(phi):
        if phi.dim() == 0:
            phi = phi.expand(batch)
        e_minus = torch.exp(-0.5j * phi)
        e_plus = torch.exp(0.5j * phi)
        gate = torch.zeros(phi.shape[0], 2, 2, device=device, dtype=dtype)
        gate[:, 0, 0] = e_minus
        gate[:, 1, 1] = e_plus
        return gate

    total = angles.shape[1]
    # Each reuploading cycle processes all data_dim features in one pass.
    chunk = total

    def encode_block(sub_angles: torch.Tensor) -> None:
        nonlocal state
        idx = 0
        if encoding in {"ryrx", "ry_rx"}:
            while idx < chunk:
                for q in range(num_qubits):
                    if idx >= chunk:
                        break
                    gate = ry_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
                for q in range(num_qubits):
                    if idx >= chunk:
                        break
                    gate = rx_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
            return
        if encoding in {"rxry", "rx_ry"}:
            while idx < chunk:
                any_applied = False
                for q in range(num_qubits):
                    if idx >= chunk:
                        break
                    gate = rx_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
                    any_applied = True
                    if idx >= chunk:
                        break
                    gate = ry_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
                if not any_applied:
                    break
                for q in range(num_qubits):
                    state = _apply_cnot(state, q, (q + 1) % num_qubits, num_qubits)
            return
        if encoding in {"rxryrz", "rx_ry_rz", "xyz"}:
            while idx < chunk:
                for q in range(num_qubits):
                    if idx >= chunk:
                        break
                    gate = rx_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
                    if idx >= chunk:
                        break
                    gate = ry_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
                    if idx >= chunk:
                        break
                    gate = rz_gate(sub_angles[:, idx]).to(device=device, dtype=dtype)
                    state = _apply_single_qubit(state, gate, q, num_qubits)
                    idx += 1
            return
        raise ValueError("encoding must be 'ryrx', 'rxry', or 'rxryrz'")

    idx_theta = 0
    for _ in range(reuploading):
        encode_block(angles)

        for _ in range(vqc_layers):
            for q in range(num_qubits):
                state = _apply_cnot(state, q, (q + 1) % num_qubits, num_qubits)
            for q in range(num_qubits):
                gate = rx_gate(theta[idx_theta]).to(device=device, dtype=dtype)
                state = _apply_single_qubit(state, gate, q, num_qubits)
                idx_theta += 1
            for q in range(num_qubits):
                gate = ry_gate(theta[idx_theta]).to(device=device, dtype=dtype)
                state = _apply_single_qubit(state, gate, q, num_qubits)
                idx_theta += 1

    probs = state.abs() ** 2  # [B, 2^n]
    idxs = torch.arange(probs.shape[1], device=device)
    values = []
    for q in range(num_qubits):
        parity = (idxs >> q) & 1
        eig = 1 - 2 * parity
        values.append((probs * eig).sum(dim=1))
    for q in range(num_qubits):
        q_next = (q + 1) % num_qubits
        parity = ((idxs >> q) ^ (idxs >> q_next)) & 1
        eig = 1 - 2 * parity
        values.append((probs * eig).sum(dim=1))
    out = torch.stack(values, dim=1)

    if return_statevector:
        return out, state
    if out.shape[0] == 1:
        return out.squeeze(0)
    return out


def measurement_dim(num_qubits: int) -> int:
    return num_qubits * 2


@dataclass
class QuantumAnsatz:
    data_dim: int
    num_qubits: int = 8
    vqc_layers: int = 1
    reuploading: int = 1
    encoding: str | None = None
    backend_device: str = "cpu"  # "cpu" or "gpu"
    use_torch_autograd: bool = True

    def __post_init__(self) -> None:
        if self.encoding is None:
            self.encoding = "rxry"
        self.encoding = self.encoding.lower()
        if self.reuploading < 1:
            raise ValueError("reuploading must be >= 1")
        self.params = None
        self.data_params = None
        self.backend = None
        self.template = None
        if not self.use_torch_autograd:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector

            self.params = ParameterVector("theta", self.param_shape)
            self.data_params = ParameterVector("x", self.data_dim)
            if self.backend_device.lower() == "gpu":
                try:
                    from qiskit_aer import AerSimulator
                except ImportError as exc:
                    raise ImportError("qiskit-aer-gpu is required for GPU backend") from exc
                self.backend = AerSimulator(method="statevector", device="GPU")
            self.template = self._build_template()

    @property
    def param_shape(self) -> int:
        return self.reuploading * self.vqc_layers * 2 * self.num_qubits

    @property
    def feature_dim(self) -> int:
        return measurement_dim(self.num_qubits)

    def _build_template(self) -> QuantumCircuit:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(self.num_qubits)
        theta_idx = 0
        for _ in range(self.reuploading):
            encode_params(
                circuit,
                self.data_params,
                num_qubits=self.num_qubits,
                encoding=self.encoding,
            )
            theta_idx = add_vqc_layers(circuit, self.params, theta_idx, self.vqc_layers, num_qubits=self.num_qubits)
        return circuit

    def circuit_for_angles(
        self, angles: Sequence[float], param_values: Sequence[float] | None = None
    ) -> QuantumCircuit:
        if self.use_torch_autograd:
            raise RuntimeError(
                "circuit_for_angles requires use_torch_autograd=False (Qiskit path). "
                "Recreate the ansatz with use_torch_autograd=False to export circuits."
            )
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        if len(angles) != self.data_dim:
            raise ValueError(f"angles length {len(angles)} does not match data_dim {self.data_dim}")
        bind = {self.data_params[i]: float(angles[i]) for i in range(self.data_dim)}
        bind.update({self.params[i]: float(param_values[i]) for i in range(self.param_shape)})
        return self.template.assign_parameters(bind, inplace=False)

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        if self.use_torch_autograd:
            angles_t = torch.as_tensor(patch_angles, dtype=torch.float32)
            theta_t = torch.as_tensor(
                param_values if param_values is not None else [0.0] * self.param_shape, dtype=torch.float32
            )
            return self.torch_features(angles_t, theta_t).detach().cpu().numpy()
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        if len(patch_angles) != self.data_dim:
            raise ValueError(f"patch_angles length {len(patch_angles)} does not match data_dim {self.data_dim}")
        bind = {self.data_params[i]: float(patch_angles[i]) for i in range(self.data_dim)}
        bind.update({self.params[i]: float(param_values[i]) for i in range(self.param_shape)})
        return z_zz_ring_features(self.template, bind, self.backend)

    def torch_features(
        self, patch_angles: torch.Tensor, theta: torch.Tensor, return_statevector: bool = False
    ) -> torch.Tensor:
        if not self.use_torch_autograd:
            if return_statevector:
                raise RuntimeError("return_statevector requires use_torch_autograd=True (torch simulation).")
            # Fallback: use numpy path (non-differentiable)
            feats = self.features(
                patch_angles.detach().cpu().numpy().tolist(),
                theta.detach().cpu().numpy().tolist(),
            )
            return torch.as_tensor(feats, dtype=theta.dtype, device=theta.device)
        return simulate_torch_features(
            patch_angles,
            theta,
            num_qubits=self.num_qubits,
            vqc_layers=self.vqc_layers,
            reuploading=self.reuploading,
            encoding=self.encoding,
            return_statevector=return_statevector,
        )


@dataclass
class QuantumPatchModel:
    patch_size: int | Sequence[int] = 4
    channels: int = 2
    num_qubits: int = 8
    vqc_layers: int = 1
    reuploading: int = 1
    encoding: str | None = None
    backend_device: str = "cpu"
    use_torch_autograd: bool = True

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
        self.params = self.ansatz.params

    @property
    def param_shape(self) -> int:
        return self.ansatz.param_shape

    def circuit_for_angles(self, angles: Sequence[float], param_values: Sequence[float] | None = None) -> QuantumCircuit:
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
            # Initialize each theta independently with Xavier-style initialization
            # Each ansatz gets different random initialization for diversity
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
        """
        Args:
            angles: [B, P, D] or [P, D]
        Returns:
            q, k, v: Each [B, P, dim]
        """
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
        """
        Process a batch of images.
        Args:
            images: [B, C, H, W]
        Returns:
            q, k, v: Each [B, num_patches, dim] tensors
        """
        if self.trainable:
            angles = torch.stack([split_patch_angles(img, patch_size) for img in images], dim=0)
            angles = angles.to(self.device or images.device)
            if return_statevector:
                return self.forward_angles(angles, param_values, return_statevector=True)
            return self.forward_angles(angles, param_values)
        if return_statevector:
            raise ValueError("return_statevector requires trainable=True.")
        batch_size = images.shape[0]
        q_list, k_list, v_list = [], [], []

        for img in images:
            q, k, v = self.forward_image(img, patch_size, param_values)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        # Stack to [B, P, D]
        q_batch = torch.stack(q_list, dim=0)
        k_batch = torch.stack(k_list, dim=0)
        v_batch = torch.stack(v_list, dim=0)

        return q_batch, k_batch, v_batch


def _stack_list(features: Sequence[np.ndarray] | Sequence[torch.Tensor], device: torch.device | None = None) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("features is empty")
    if isinstance(features[0], torch.Tensor):
        return torch.stack([f.to(device=device) for f in features], dim=0)
    return torch.stack([torch.from_numpy(np.asarray(f)).to(device=device) for f in features], dim=0)


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
                q_ansatz = _clone_ansatz(ansatz_q, hidden_dim)
                k_ansatz = _clone_ansatz(ansatz_k, hidden_dim)
                v_ansatz = _clone_ansatz(ansatz_v, hidden_dim)
                self.qkv_layers.append(
                    SeparateQKVProjector(
                        ansatz_q=q_ansatz,
                        ansatz_k=k_ansatz,
                        ansatz_v=v_ansatz,
                        device=self.device,
                        trainable=True,
                    )
                )
        in_dim = self.attn_dim * self.patch_count
        self.classifier = ClassifierHead(in_dim=in_dim, out_dim=num_classes)
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
        entry = {
            "epoch": self.current_epoch,
            "q": q_sv.detach().cpu(),
            "k": k_sv.detach().cpu(),
            "v": v_sv.detach().cpu(),
        }
        self.saved_statevectors.setdefault(layer_idx, []).append(entry)

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

        # Compute attention statistics from first sample in batch
        attn_stats = None
        if weights_list:
            w = weights_list[0]  # [B, P, P]
            with torch.no_grad():
                # Average over batch
                entropy = -(w * (w + 1e-12).log()).sum(dim=-1).mean().item()
                max_w = w.max().item()
            attn_stats = {"entropy": entropy, "max_weight": max_w}

        # Aggregation
        emb = x.flatten(start_dim=1)  # [B, P*D]
        
        # Classification
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
        # q, k, v: [B, P, D]
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        # squared euclidean distances
        q_exp = q.unsqueeze(-2)  # [B, P, 1, D]
        k_exp = k.unsqueeze(-3)  # [B, 1, P, D]
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
