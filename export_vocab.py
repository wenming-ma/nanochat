"""
导出 tokenizer 词汇表到文本文件，方便查看
"""
from nanochat.tokenizer import get_tokenizer
import os

# 加载 tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,} tokens")

# 导出路径
output_file = "vocab_export.txt"
print(f"Exporting vocabulary to: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    # 写入头部信息
    f.write("=" * 100 + "\n")
    f.write("Nanochat Tokenizer Vocabulary\n")
    f.write(f"Total tokens: {vocab_size:,}\n")
    f.write("=" * 100 + "\n\n")

    # 特殊 tokens
    f.write("🔖 SPECIAL TOKENS\n")
    f.write("-" * 100 + "\n")
    special_tokens = tokenizer.get_special_tokens()
    for special in special_tokens:
        token_id = tokenizer.encode_special(special)
        f.write(f"ID {token_id:6d}: {special}\n")
    f.write("\n")

    # 单字节 tokens (0-255)
    f.write("📝 SINGLE BYTE TOKENS (0-255)\n")
    f.write("-" * 100 + "\n")
    for i in range(256):
        try:
            decoded = tokenizer.decode([i])
            # 转义特殊字符
            if decoded.isprintable() and decoded not in [' ', '\t', '\n', '\r']:
                repr_str = decoded
            else:
                repr_str = repr(decoded)
            f.write(f"ID {i:6d}: {repr_str:20s}  (byte: {i:3d} = 0x{i:02X})\n")
        except:
            f.write(f"ID {i:6d}: <decode error>\n")
    f.write("\n")

    # Merged tokens (256 到 vocab_size-1)
    f.write(f"🔤 MERGED TOKENS (256 - {vocab_size-1})\n")
    f.write("-" * 100 + "\n")
    f.write("Showing first 500 and last 100 merged tokens...\n\n")

    # 前 500 个 merged tokens
    for i in range(256, min(756, vocab_size)):
        try:
            decoded = tokenizer.decode([i])
            # 限制显示长度
            if len(decoded) > 50:
                decoded_show = decoded[:47] + "..."
            else:
                decoded_show = decoded
            f.write(f"ID {i:6d}: {repr(decoded_show)}\n")
        except:
            f.write(f"ID {i:6d}: <decode error>\n")

    if vocab_size > 756:
        f.write(f"\n... (skipped {vocab_size - 856} tokens) ...\n\n")

        # 最后 100 个 tokens
        for i in range(max(256, vocab_size - 100), vocab_size):
            try:
                decoded = tokenizer.decode([i])
                if len(decoded) > 50:
                    decoded_show = decoded[:47] + "..."
                else:
                    decoded_show = decoded

                # 标记特殊 token
                is_special = decoded in special_tokens
                marker = "⭐" if is_special else "  "
                f.write(f"{marker} ID {i:6d}: {repr(decoded_show)}\n")
            except:
                f.write(f"   ID {i:6d}: <decode error>\n")

print(f"\n✅ Vocabulary exported to: {os.path.abspath(output_file)}")
print(f"   Total: {vocab_size:,} tokens")
print(f"\n💡 Use any text editor to view the file!")
print(f"   Example: notepad {output_file}")
