"""
查看 nanochat tokenizer 的词汇表
"""
import pickle
import os

# 加载 tokenizer
tokenizer_path = os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl")
print(f"Loading tokenizer from: {tokenizer_path}\n")

with open(tokenizer_path, "rb") as f:
    enc = pickle.load(f)

# 获取词汇表信息
vocab_size = enc.n_vocab
print(f"词汇表大小: {vocab_size:,} tokens\n")
print("=" * 80)

# 查看特殊 tokens
print("\n🔖 特殊 Tokens:")
print("-" * 80)
special_tokens = enc._special_tokens
for token_str, token_id in sorted(special_tokens.items(), key=lambda x: x[1]):
    print(f"ID {token_id:6d}: {token_str}")

# 查看前 100 个基础 tokens (0-99: 单字节)
print("\n📝 前 100 个 Tokens (单字节 0-99):")
print("-" * 80)
for i in range(0, 100, 10):
    tokens = []
    for j in range(i, min(i+10, 100)):
        try:
            decoded = enc.decode([j])
            # 转义不可见字符
            if decoded.isprintable():
                repr_str = decoded
            else:
                repr_str = repr(decoded)[1:-1]  # 去掉引号
            tokens.append(f"{j:3d}:{repr_str:5s}")
        except:
            tokens.append(f"{j:3d}:???")
    print(" | ".join(tokens))

# 查看一些常见的 merged tokens
print("\n🔤 示例 Merged Tokens (256-356):")
print("-" * 80)
for i in range(256, 357, 10):
    tokens = []
    for j in range(i, min(i+10, 357)):
        try:
            decoded = enc.decode([j])
            # 只显示可打印的
            if decoded.isprintable() and len(decoded) <= 20:
                tokens.append(f"{j}:{decoded}")
        except:
            pass
    if tokens:
        print(" | ".join(tokens))

# 查看最后 20 个 tokens (包括特殊 tokens)
print("\n🎯 最后 20 个 Tokens:")
print("-" * 80)
for i in range(vocab_size - 20, vocab_size):
    try:
        decoded = enc.decode([i])
        # 检查是否是特殊 token
        is_special = decoded in special_tokens
        marker = "⭐" if is_special else "  "
        print(f"{marker} ID {i:6d}: {repr(decoded)}")
    except:
        print(f"   ID {i:6d}: <decode error>")

# 交互式查询
print("\n" + "=" * 80)
print("\n💡 你可以用这个脚本查看特定 token:")
print("\n示例代码:")
print("""
import pickle
with open(r"~/.cache/nanochat/tokenizer/tokenizer.pkl", "rb") as f:
    enc = pickle.load(f)

# 编码文本
token_ids = enc.encode("Hello world!")
print(f"Token IDs: {token_ids}")

# 查看每个 token
for token_id in token_ids:
    decoded = enc.decode([token_id])
    print(f"  {token_id:5d} -> {repr(decoded)}")
""")
