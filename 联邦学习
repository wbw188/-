import os
import time
import base64
import hashlib
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 密码学库
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# 同态加密
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("警告: tenseal 未安装，同态加密功能将被禁用。")

# ---------------------------- 辅助函数 ----------------------------
def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """将PyTorch张量转换为字节串（保留形状和类型信息通过元数据）"""
    return t.cpu().detach().numpy().tobytes()

def _bytes_to_tensor(b: bytes, shape: tuple, dtype_str: str) -> torch.Tensor:
    """将字节串转换回PyTorch张量"""
    # 解析dtype
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
        'torch.float': torch.float32,
        'torch.int': torch.int32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    # 从字节构造numpy数组再转tensor
    arr = np.frombuffer(b, dtype=dtype_map_inv(dtype_str))
    return torch.tensor(arr, dtype=dtype).reshape(shape)

def dtype_map_inv(dtype_str: str) -> np.dtype:
    """辅助函数，将torch dtype字符串映射为numpy dtype"""
    m = {
        'torch.float32': np.float32,
        'torch.float64': np.float64,
        'torch.int32': np.int32,
        'torch.int64': np.int64,
    }
    return m.get(dtype_str, np.float32)

# ---------------------------- 加密算法类 ----------------------------
class SymmetricEncryption:
    def __init__(self, password="federated_learning_key"):
        self.key = hashlib.sha256(password.encode()).digest()[:32]
        self.backend = default_backend()
        self.padding = sym_padding.PKCS7(128)

    def encrypt_tensor(self, t: torch.Tensor) -> Dict:
        tb = _tensor_to_bytes(t)
        iv = os.urandom(16)
        padder = self.padding.padder()
        padded = padder.update(tb) + padder.finalize()
        encryptor = Cipher(algorithms.AES(self.key), modes.CBC(iv), self.backend).encryptor()
        enc_data = encryptor.update(padded) + encryptor.finalize()
        meta = {
            'shape': t.shape,
            'dtype': str(t.dtype),
            'iv': base64.b64encode(iv).decode('utf-8')
        }
        return {'data': enc_data, 'metadata': meta}

    def decrypt_tensor(self, pkg: Dict) -> torch.Tensor:
        enc_data = pkg['data']
        meta = pkg['metadata']
        iv = base64.b64decode(meta['iv'].encode('utf-8'))
        decryptor = Cipher(algorithms.AES(self.key), modes.CBC(iv), self.backend).decryptor()
        padded = decryptor.update(enc_data) + decryptor.finalize()
        unpadder = self.padding.unpadder()
        tb = unpadder.update(padded) + unpadder.finalize()
        return _bytes_to_tensor(tb, meta['shape'], meta['dtype'])


class AsymmetricEncryption:
    def __init__(self):
        key_size = 2048
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.chunk_size = 128  # RSA加密分块大小（OAEP填充后实际加密数据长度小于密钥长度）
        self.padding = asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )

    def encrypt_tensor(self, t: torch.Tensor) -> Dict:
        tb = _tensor_to_bytes(t)
        chunks = []
        for i in range(0, len(tb), self.chunk_size):
            chunk = tb[i:i+self.chunk_size]
            encrypted_chunk = self.public_key.encrypt(chunk, self.padding)
            chunks.append(encrypted_chunk)
        return {
            'data': chunks,
            'metadata': {'shape': t.shape, 'dtype': str(t.dtype)}
        }

    def decrypt_tensor(self, pkg: Dict) -> torch.Tensor:
        chunks = pkg['data']
        meta = pkg['metadata']
        decrypted_bytes = b''.join([self.private_key.decrypt(c, self.padding) for c in chunks])
        return _bytes_to_tensor(decrypted_bytes, meta['shape'], meta['dtype'])


class DifferentialPrivacyEncryption:
    def __init__(self):
        self.clip_norm = 1.5
        self.epsilon = 16.0
        self.delta = 1e-5
        noise_multiplier = 0.12
        self.noise_scale = (self.clip_norm * noise_multiplier *
                            np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon

    def encrypt_tensor(self, t: torch.Tensor) -> Dict:
        norm = torch.norm(t).item()
        clipped = t * min(1.0, self.clip_norm / (norm + 1e-10))
        private = clipped + torch.randn_like(clipped) * self.noise_scale
        return {'data': private, 'metadata': {'shape': t.shape, 'dtype': str(t.dtype)}}

    def decrypt_tensor(self, pkg: Dict) -> torch.Tensor:
        return pkg['data']

    def get_privacy_spent(self, n_samples, b_size, epochs):
        if n_samples == 0:
            return 0.0
        steps = max(1, n_samples // b_size) * epochs
        return min(self.epsilon * np.sqrt(steps), self.epsilon * np.sqrt(steps))


class HomomorphicEncryption:
    def __init__(self):
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("tenseal 库未安装，无法使用同态加密。")
        ctx_args = {
            'scheme': ts.SCHEME_TYPE.CKKS,
            'poly_modulus_degree': 16384,
            'coeff_mod_bit_sizes': [60, 40, 40, 40, 40, 60]
        }
        self.context = ts.context(**ctx_args)
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

    def encrypt_tensor(self, t: torch.Tensor) -> Dict:
        enc_vec = ts.ckks_vector(self.context, t.flatten().tolist())
        return {'data': enc_vec, 'metadata': {'shape': t.shape, 'dtype': str(t.dtype)}}

    def decrypt_tensor(self, pkg: Dict) -> torch.Tensor:
        meta = pkg['metadata']
        dec_list = pkg['data'].decrypt()
        dtype_str = meta['dtype']
        # 解析dtype
        dtype_map = {'torch.float32': torch.float32, 'torch.float64': torch.float64}
        dtype = dtype_map.get(dtype_str, torch.float32)
        return torch.tensor(dec_list, dtype=dtype).reshape(meta['shape'])


# ---------------------------- 模型定义 ----------------------------
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


# ---------------------------- 联邦学习客户端 & 服务器 ----------------------------
def train_client(model, loader, criterion, optimizer, encryptor, max_batches=5):
    """客户端训练：计算最后一个batch的梯度并用加密器加密后返回"""
    model.train()
    enc_grads = []
    shapes = []
    batches_done = 0
    for images, labels in loader:
        if batches_done >= max_batches:
            break
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        batches_done += 1
        if batches_done == max_batches:
            for p in model.parameters():
                grad = p.grad.clone() if p.grad is not None else None
                if grad is not None:
                    enc_grads.append(encryptor.encrypt_tensor(grad))
                else:
                    enc_grads.append(None)
                shapes.append(tuple(p.shape))
            break
    if not any(g is not None for g in enc_grads):
        return None, None
    return enc_grads, shapes


def server_aggregate(model, enc_grads_list, shapes, encryptor, optimizer, enc_type):
    """服务器聚合加密梯度并更新模型"""
    num_clients = len(enc_grads_list)
    if num_clients == 0:
        return model
    num_params = len(shapes)
    agg_grads = [None] * num_params

    with torch.no_grad():
        for p_idx in range(num_params):
            valid_pkgs = []
            for client_grads in enc_grads_list:
                if client_grads and p_idx < len(client_grads) and client_grads[p_idx] is not None:
                    valid_pkgs.append(client_grads[p_idx])
            n_valid = len(valid_pkgs)
            if n_valid == 0:
                continue

            agg_grad = None
            try:
                if enc_type in ["symmetric", "asymmetric", "differential_privacy"]:
                    decrypted = [encryptor.decrypt_tensor(pkg) for pkg in valid_pkgs]
                    valid_dec = [g for g in decrypted if g is not None]
                    if valid_dec:
                        agg_grad = sum(valid_dec) / len(valid_dec)
                elif enc_type == "homomorphic":
                    # 密文直接相加
                    agg_enc = valid_pkgs[0]['data'].copy()
                    for i in range(1, n_valid):
                        agg_enc += valid_pkgs[i]['data']
                    pkg = {'data': agg_enc, 'metadata': valid_pkgs[0]['metadata']}
                    summed = encryptor.decrypt_tensor(pkg)
                    if summed is not None:
                        agg_grad = summed / n_valid
            except Exception as e:
                print(f"聚合参数 {p_idx} 时出错: {e}")

            if agg_grad is not None and tuple(agg_grad.shape) == shapes[p_idx]:
                agg_grads[p_idx] = agg_grad

        # 应用梯度
        applied = 0
        for param, grad in zip(model.parameters(), agg_grads):
            if param.requires_grad and grad is not None:
                param.grad = grad
                applied += 1
        if applied > 0:
            optimizer.step()
    return model


def train_plaintext(model, loader, criterion, optimizer, total_batches):
    """明文训练（普通SGD）"""
    model.train()
    start = time.time()
    batches = 0
    for images, labels in loader:
        if batches >= total_batches:
            break
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batches += 1
    return time.time() - start


def train_encrypted(model, loader, criterion, optimizer, encryptor, enc_type, num_rounds,
                    num_clients=3, client_batches=5):
    """加密联邦学习训练"""
    start = time.time()
    for r in range(num_rounds):
        round_grads = []
        current_shapes = None
        successful = 0
        for c in range(num_clients):
            grads, shapes = train_client(model, loader, criterion, optimizer, encryptor, client_batches)
            if grads and shapes:
                if current_shapes is None:
                    current_shapes = shapes
                if shapes == current_shapes:
                    round_grads.append(grads)
                    successful += 1
        if successful > 0 and current_shapes:
            server_aggregate(model, round_grads, current_shapes, encryptor, optimizer, enc_type)
        # 差分隐私隐私预算打印
        if isinstance(encryptor, DifferentialPrivacyEncryption):
            total_samples = len(loader.dataset)
            steps = num_rounds * num_clients * client_batches
            eps = encryptor.get_privacy_spent(total_samples, loader.batch_size, steps)
            print(f"  轮次 {r+1}: 累计隐私预算 ε ≈ {eps:.4f}")
    return time.time() - start


def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def create_encryptor(enc_type: str):
    """根据字符串创建对应的加密器实例"""
    if enc_type == "symmetric":
        return SymmetricEncryption()
    elif enc_type == "asymmetric":
        return AsymmetricEncryption()
    elif enc_type == "differential_privacy":
        return DifferentialPrivacyEncryption()
    elif enc_type == "homomorphic":
        if not TENSEAL_AVAILABLE:
            raise ImportError("tenseal not installed")
        return HomomorphicEncryption()
    else:
        raise ValueError(f"Unknown encryption type: {enc_type}")


# ---------------------------- 主实验 ----------------------------
def run_experiment(model_class, model_name, trainloader, testloader, num_rounds=3,
                   num_clients=3, client_batches=5):
    print(f"\n{'='*60}")
    print(f"实验模型: {model_name}")
    print(f"联邦学习配置: 轮数={num_rounds}, 客户端数={num_clients}, 每客户端批次数={client_batches}")
    print(f"明文训练总批次数 = {num_rounds * num_clients * client_batches}")
    print('='*60)

    # 明文训练
    model_plain = model_class()
    criterion = nn.CrossEntropyLoss()
    optimizer_plain = torch.optim.SGD(model_plain.parameters(), lr=0.01)
    total_batches = num_rounds * num_clients * client_batches
    time_plain = train_plaintext(model_plain, trainloader, criterion, optimizer_plain, total_batches)
    acc_plain = evaluate_model(model_plain, testloader)
    print(f"\n[明文] 准确率: {acc_plain:.2f}% , 训练时间: {time_plain:.2f} 秒")

    # 不同加密方法
    enc_types = ["symmetric", "asymmetric", "differential_privacy"]
    if TENSEAL_AVAILABLE:
        enc_types.append("homomorphic")
    results = {}

    for enc_type in enc_types:
        print(f"\n--- 加密类型: {enc_type} ---")
        model_enc = model_class()
        optimizer_enc = torch.optim.SGD(model_enc.parameters(), lr=0.01)
        encryptor = create_encryptor(enc_type)
        try:
            time_enc = train_encrypted(model_enc, trainloader, criterion, optimizer_enc,
                                       encryptor, enc_type, num_rounds, num_clients, client_batches)
            acc_enc = evaluate_model(model_enc, testloader)
            results[enc_type] = (acc_enc, time_enc)
            print(f"准确率: {acc_enc:.2f}% , 训练时间: {time_enc:.2f} 秒")
        except Exception as e:
            print(f"训练失败: {e}")
            results[enc_type] = (0.0, 0.0)

    # 汇总表格
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    print(f"{'方法':<20} {'准确率(%)':<15} {'训练时间(秒)':<15}")
    print("-"*50)
    print(f"{'明文 (无加密)':<20} {acc_plain:<15.2f} {time_plain:<15.2f}")
    for enc, (acc, t) in results.items():
        name = enc.replace('_', ' ').title()
        print(f"{name:<20} {acc:<15.2f} {t:<15.2f}")
    print("="*60)


if __name__ == "__main__":
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    # 运行实验（使用 LinearClassifier 和 SimpleCNN 两种模型）
    run_experiment(LinearClassifier, "LinearClassifier", trainloader, testloader,
                   num_rounds=3, num_clients=3, client_batches=5)

    # 如果计算资源允许，可以运行CNN模型（需要更多时间）
    # run_experiment(SimpleCNN, "SimpleCNN", trainloader, testloader,
    #                num_rounds=3, num_clients=3, client_batches=5)
