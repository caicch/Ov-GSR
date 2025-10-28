# import google.generativeai as genai
# # 配置API密钥和传输协议
# genai.configure(api_key="AIzaSyCjHeBiXKBiiHyS_cPKoDU4shMyB4b4qz0", transport='rest')

# 初始化模型并生成内容
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("请使用中文，解释你是谁")
# print(response.text)

import google.generativeai as genai

print("2")
# 配置API密钥和传输协议
try:
    genai.configure(api_key="AIzaSyCaqAJVGgUPwtYTMoYPm7u9bI9rdqs3GYc", transport='rest')
    print("API配置成功")
except Exception as e:
    print(f"API配置失败: {e}")

print("3")
# 初始化模型并生成内容
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("模型初始化成功")
except Exception as e:
    print(f"模型初始化失败: {e}")

print("4")
# 生成内容
try:
    response = model.generate_content("请使用中文，柴犬好还是萨摩耶好？")
    print("生成内容成功")
    print(response.text)
except Exception as e:
    print(f"生成内容失败: {e}")
