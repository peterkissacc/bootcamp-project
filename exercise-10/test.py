import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI

# 1. Töltsük be a változókat
loaded = load_dotenv(find_dotenv(), override=True)
print(f"Környezeti változók betöltve: {loaded}")

# 2. Ellenőrizzük, hogy a kulcsok látszódnak-e
api_key = os.getenv("LANGCHAIN_API_KEY")
project = os.getenv("LANGCHAIN_PROJECT")
print(f"LangSmith Project: {project}")
print(f"LangSmith API Key (első 4 karakter): {api_key[:4] if api_key else 'NINCS MEGADVA'}")

# 3. Futtassunk egy egyszerű hívást
# (Itt felhasználjuk a .env-ben lévő Azure beállításokat is)
try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    print("Hívás küldése a LangSmith felé...")
    response = llm.invoke("Mondd, hogy 'Hello LangSmith'!")
    print(f"Válasz: {response.content}")
    print("\n✅ SIKER! Most frissítsd a LangSmith weboldalt.")

except Exception as e:
    print(f"\n❌ HIBA TÖRTÉNT: {e}")