@echo off

cd /d C:\Users\Kavya\Downloads\mashreq_neo\mashreq-neo-rag-chatbot

call .venv\Scripts\activate.bat

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause