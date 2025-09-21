import os
import telebot
from telebot import types
import re
import requests
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Инициализация бота
TELEGRAM_BOT_TOKEN = ""  # Замените на ваш токен
GROQ_API_KEY = ""  # Замените на ваш Groq API-ключ
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


# Словарь для хранения состояния пользователей
user_data = {}

# Функция для анализа симптомов
def analyze_symptoms(query, disease_list):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    prompt_template = """
    Ты — медицинский эксперт. Проанализируй следующие симптомы: {query}.
    (Если этот список не содержит симптомы, то выведи "False". Не включай в свой ответ дополнительные описания или пояснения.)
    Выдели из предложенного списка заболеваний топ 7 заболеваний, которые чаще всего встречаются с такими симптомами.
    Убедись, что заболеваний не повторяются.
    Не включай в свой ответ дополнительные описания или пояснения.
    Выведи спиcок заболеваний через запятую.
    (Пример ответа: Острое респираторное заболевание (ОРЗ), Ринит, Рипофарингит, Острый ларинго-трахеобронхит)
    Список заболеваний: {context}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
    chain = LLMChain(prompt=prompt, llm=llm)
    context = ", ".join(disease_list)
    response = chain.run(query=query, context=context)
    return response

# Функция для поиска симптомов для заболеваний
def find_symptoms(response, query):
    text_loader = TextLoader("книга.txt", encoding='UTF-8')
    documents = text_loader.load()
    text_contents = documents[0].page_content
    paragraphs = [value for value in text_contents.split("\n\n") if value.strip()]
    split_documents = [Document(page_content=text) for text in paragraphs]
    emb_model2 = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_storeIM2 = FAISS.from_documents(split_documents, emb_model2)

    maybe = {}
    sympt = []
    for name in response[:-1].split(", "):
        if name in disease_list:
            answer = vector_storeIM2.similarity_search(f"Какие симптомы у заболевания {name}?", k=2)
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
            prompt_template = """
            Ты — медицинский эксперт. Проанализируй следующие симптомы заболевания {name}: {context}.
            Ответь "YES", если среди симптомов этого заболевания присутствуют: {query}, или схожие с ними симптомы.
            После этого выведи список всех симптомов этого заболевания в именительном падеже через запятую.
            Если этих симптомов нет, то ответь "NO".
            Не включай в свой ответ дополнительные описания или пояснения.
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["name", "context", "query"])
            chain = LLMChain(prompt=prompt, llm=llm)
            response_v = chain.run(name=name, context="\n".join([doc.page_content for doc in answer]), query=query)
            if "YES" in response_v:
                tmp_str = response_v.strip("YES")
                if ":" in tmp_str:
                    tmp_str = tmp_str[tmp_str.index(":") + 1:]
                tmp_str = tmp_str.split(", ")
                sympt += tmp_str
                maybe[name] = tmp_str
    sympt = list(dict.fromkeys(sympt))
    return maybe, sympt

# Функция для генерации вопросов для уточнения диагноза
def symptoms(sympt):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    prompt_template = """
    Ты — медицинский эксперт. Проанализируй следующие возможные симптомы заболеваний {context}.
    Выдели из этих симптомов 5 различных, которые помогут максимально сузить круг возможных заболеваний.
    Убедись, что эти симптомы не синонимичны и достаточно отличаются по смыслу.
    Дай каждому из симптомов простое и понятное определение на русском языке.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = LLMChain(prompt=prompt, llm=llm)
    response_3 = chain.run(context=sympt)
    return response_3

# Функция для обработки ответов пользователя
def answers(sympt_checked, response_3, sympt, maybe, chat_id):
    dis_ans = {key: [] for key in maybe.keys()}
    i = 0
    for s in sympt:
        if s.capitalize() in response_3:
            ans = user_data[chat_id].get("last_answer", "нет")
            for d in maybe.keys():
                if s in maybe[d]:
                    if ans == "да":
                        dis_ans[d].append(1)
                    elif ans == "нет":
                        dis_ans[d].append(0)
            sympt_checked.append(s)
            i += 1
            if i == 10:
                break
    return dis_ans, sympt_checked

# Функция для формирования окончательного диагноза
def disease(sympt, maybe, chat_id):
    sympt_checked = []
    dis_ans = {}
    x = list(set(sympt) - set(sympt_checked))
    flag = True
    while flag:
        response_3 = symptoms(x)
        dis_ans, sympt_checked = answers(sympt_checked, response_3, sympt, maybe, chat_id)
        ans = {}
        for key in dis_ans.keys():
            count = sum(True for i in dis_ans[key] if i == 1)
            ans[key] = count
        sorted_people = sorted(ans.items(), key=lambda item: item[1])
        sorted_people.reverse()
        if sorted_people[0][1] >= sorted_people[2][1] + 2:
            flag = False
        else:
            tmp_x = []
            for i in sorted_people:
                if sorted_people[0][1] < i[1] + 2:
                    for s in dis_ans[i[0]]:
                        if s not in x and s not in sympt_checked:
                            tmp_x.append(s)
            x = tmp_x
            if x == []:
                flag = False
    return sorted_people[0][0], sorted_people[1][0], sorted_people[2][0]

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    user_data[chat_id] = {"state": "awaiting_symptoms"}
    bot.send_message(chat_id, "Введите ваши симптомы:")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_state = user_data.get(chat_id, {}).get("state")

    if user_state == "awaiting_symptoms":
        # Пользователь ввел симптомы
        symptoms = message.text.strip()
        if not symptoms:
            bot.send_message(chat_id, "Пожалуйста, опишите свои симптомы.")
            return

        # Анализ симптомов
        response = analyze_symptoms(symptoms, disease_list)
        if response == "False":
            bot.send_message(chat_id, "Не удалось определить заболевания по вашим симптомам.")
            return

        # Поиск симптомов для заболеваний
        maybe, sympt = find_symptoms(response, symptoms)
        if not sympt:
            bot.send_message(chat_id, "Не удалось найти симптомы для заболеваний.")
            return

        # Сохраняем состояние пользователя
        user_data[chat_id]["symptoms"] = symptoms
        user_data[chat_id]["maybe"] = maybe
        user_data[chat_id]["sympt"] = sympt
        user_data[chat_id]["sympt_checked"] = []
        user_data[chat_id]["state"] = "clarifying_symptoms"
        user_data[chat_id]["current_symptom_index"] = 0

        # Задаем первый вопрос
        ask_next_question(chat_id)
    elif user_state == "clarifying_symptoms":
        answer = message.text.strip().lower()
        user_data[chat_id]["last_answer"] = answer

        # Обновляем состояние пользователя
        current_symptom_index = user_data[chat_id]["current_symptom_index"]
        sympt = user_data[chat_id]["sympt"]
        sympt_checked = user_data[chat_id]["sympt_checked"]
        maybe = user_data[chat_id]["maybe"]

        if current_symptom_index >= len(sympt):
            finalize_diagnosis(chat_id)
            return

        symptom = sympt[current_symptom_index]
        for d in maybe.keys():
            if symptom in maybe[d]:
                if answer == "да":
                    user_data[chat_id]["maybe"][d].append(1)
                elif answer == "нет":
                    user_data[chat_id]["maybe"][d].append(0)
        sympt_checked.append(symptom)
        user_data[chat_id]["sympt_checked"] = sympt_checked

        # Переходим к следующему вопросу
        user_data[chat_id]["current_symptom_index"] += 1
        ask_next_question(chat_id)


def ask_next_question(chat_id):
    if "current_symptom_index" not in user_data[chat_id]:
        user_data[chat_id]["current_symptom_index"] = 0

    current_index = user_data[chat_id]["current_symptom_index"]
    sympt = user_data[chat_id]["sympt"]
    sympt_checked = user_data[chat_id]["sympt_checked"]

    if current_index >= len(sympt):
        finalize_diagnosis(chat_id)
        return

    symptom = sympt[current_index]
    user_data[chat_id]["current_symptom"] = symptom
    bot.send_message(chat_id, f"Есть ли у вас симптом: {symptom}? (ответьте 'да' или 'нет')")


def finalize_diagnosis(chat_id):
    sympt = user_data[chat_id]["sympt"]
    maybe = user_data[chat_id]["maybe"]
    sympt_checked = user_data[chat_id]["sympt_checked"]

    # Формируем окончательный диагноз
    diag1, diag2, diag3 = disease(sympt, maybe, chat_id)
    bot.send_message(chat_id, f"Топ 3 возможных заболевания:\n\n{diag1}:\n\n" +
    f"{description(diag1)}\n\n{diag2}:\n\n{description(diag2)}" +
                     f"\n\n{diag3}:\n\n{description(diag3)}")

    # Сбрасываем состояние пользователя
    user_data[chat_id] = {}


def description(diag):
    text_loader = TextLoader("книга.txt", encoding='UTF-8')
    documents = text_loader.load()
    text_contents = documents[0].page_content
    paragraphs = [value for value in text_contents.split("\n\n") if value.strip()]
    split_documents = [Document(page_content=text) for text in paragraphs]
    emb_model2 = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_storeIM2 = FAISS.from_documents(split_documents, emb_model2)

    answer = vector_storeIM2.similarity_search(f"{diag} - это", k=2)
    answer2 = vector_storeIM2.similarity_search(f"Как лечится {name}?", k=2)

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    prompt_template = """
    Ты — медицинский эксперт. Объясни, что такое {name}, на основе следующей информации:
    {context}
    Укажи основные методы лечения и рекомендации при болезни.
    {context2}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["name", "context", "context2"])
    chain = LLMChain(prompt=prompt, llm=llm)
    response_descr = chain.run(name=diag, context=answer, context2=answery2)

    return response_descr




# Запуск бота
if __name__ == "__main__":
    bot.polling()
