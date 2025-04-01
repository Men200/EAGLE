import json

file_path = "/home/xwwen/EAGLE_test/ShareGPT_dataset/computer_en_26k.jsonl"
# output_path = "/home/xwwen/EAGLE_test/ShareGPT_dataset/converted.json"

# # 读取 JSONL 文件并转换为 JSON
# with open(file_path, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# # 转换成 JSON 格式
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump({"data": data}, f, indent=4, ensure_ascii=False)

# print("转换完成，已保存为 converted.json")

output_file = "/home/xwwen/EAGLE_test/ShareGPT_dataset/cleaned.jsonl"
error_cnt = 0
with open(file_path, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            data = json.loads(line)
            # 确保 category 是字符串，如果是列表则提取第一个元素
            if isinstance(data.get("category"), list):
                error_cnt += 1
                data["category"] = data["category"][0] if data["category"] else "Unknown"
            elif not isinstance(data.get("category"), str):
                data["category"] = str(data["category"])
            fout.write(json.dumps(data) + "\n")
        except json.JSONDecodeError:
            print("Skipping invalid line:", line.strip())
print(f"总计存在问题的数据条数：{error_cnt}")