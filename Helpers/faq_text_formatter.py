import json

with open("faq.txt") as file:
    formatter = []
    sections = json.load(file)
    for section in sections:
        for value in sections[section]:
            value_formatter = {
              "tag": section.lower(),
              "patterns": [
                value["faq_question"]
              ],
              "responses": [
                value["faq_answer"]
              ],
            }
            formatter.append(value_formatter)
    out_file = open("faq_formatted.json", "w")
    json.dump(formatter, out_file)
    out_file.close()

with open("faq.txt") as file:
    formatter = ""
    sections = json.load(file)
    for section in sections:
        for value in sections[section]:
            value_formatter = f"{value['faq_question']} {value['faq_answer']}"
            formatter += value_formatter + " "
    out_file = open("faq_text.json", "w")
    json.dump(formatter, out_file)
    out_file.close()