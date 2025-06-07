# 1. ابدأ من نسخة بايثون 3.10
FROM python:3.10

# 2. حدد مجلد العمل جوه السيرفر
WORKDIR /code

# 3. انسخ ملف المكتبات الأول عشان نثبتهم
COPY ./requirements.txt /code/requirements.txt

# 4. شغل أمر تثبيت المكتبات
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. انسخ كل ملفات مشروعك الباقية
COPY . /code/

# 6. افتح "بوابة" رقم 7860
EXPOSE 7860

# 7. الأمر النهائي اللي هيشغل تطبيقك
CMD ["python", "my_flask_app.py"]