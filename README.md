# 🤖 Assistant Documentaire Intelligent (RAG)

Une application web interactive construite avec **Streamlit** et **LangChain** qui permet de discuter avec des documents PDF en utilisant une architecture de génération augmentée par récupération (RAG).

## 🌟 Fonctionnalités
* **Chargement de PDF** : Extraction de texte intelligente via `PyPDFLoader`.
* **Traitement de texte** : Découpage optimisé avec `RecursiveCharacterTextSplitter`.
* **Embeddings de pointe** : Utilisation des modèles HuggingFace pour transformer le texte en vecteurs.
* **Base de données vectorielle** : Stockage local rapide avec `FAISS`.
* **LLM Local** : Intégration avec `Ollama` (ChatOllama) pour une confidentialité totale des données.

## 🛠️ Installation

1.  **Cloner le projet :**
    ```bash
    git clone [https://github.com/ton-pseudo/nom-du-repo.git](https://github.com/ton-pseudo/nom-du-repo.git)
    cd nom-du-repo
    ```

2.  **Créer un environnement virtuel :**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration locale :**
    Assurez-vous qu'Ollama est installé et qu'un modèle (ex: `llama3` ou `mistral`) est en cours d'exécution.J'ai utilisé gemma:2b qui est un modèle léger.

## 🚀 Utilisation

Lancez l'application avec la commande suivante :
```bash
streamlit run app.py
