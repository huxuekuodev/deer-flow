import xml.etree.ElementTree as etree

if __name__ == "__main__":
    content = "<current_time>2026-07-24</current_time>"
    isExits = False
    if content.strip().startswith("<current_time>"):
        try:
            current_time = etree.fromstring(content)
            # 将current_time text 内容转时间戳，格式为yyyy-MM-dd 验证是否是今日

            import datetime

            parsed_date = datetime.datetime.strptime(current_time.text, "%Y-%m-%d").date()
            if parsed_date == datetime.date.today():
                print("是今日")
            else:
                print("不是今日")

            isExits = True
        except Exception:
            pass
