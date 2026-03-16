import ssl
import os
# Hack pour contourner les erreurs SSL sur les réseaux sécurisés/Windows
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
import sentence_transformers
import streamlit as st
import os
import tempfile

# Les chargeurs et splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Les modèles et vecteurs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

# Les composants pour la chaîne
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure le titre de l'onglet du navigateur et l'icône
st.set_page_config(page_title="Light RAG ENSI", page_icon="⚡")

# Titre affiché sur la page
st.title("⚡ Assistant de Révision (Mode Léger)")

# Initialisation de l'historique du chat dans la mémoire de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Création de la barre latérale (sidebar) à gauche
with st.sidebar:
    st.header("1. Chargez votre cours")
    
    # Widget pour uploader le fichier PDF
    uploaded_file = st.file_uploader("Fichier PDF", type="pdf")

# Si un fichier est chargé, on lance le traitement
if uploaded_file is not None:
    
    # ÉTAPE A : Création d'un fichier physique temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # ÉTAPE B : Indexation (On vérifie si on l'a déjà fait pour ne pas perdre de temps)
    if "vectorstore" not in st.session_state:
        with st.spinner("Analyse légère en cours..."):
            # 1. Chargement du document depuis le chemin temporaire
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 2. Découpage (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            # 3. Création des Embeddings
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
)

            # 4. Création de la base vectorielle (FAISS) et stockage en session
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vectorstore = vectorstore
            
            st.success("Cours prêt ! (Indexé en mémoire)")
            
            # Nettoyage : On supprime le fichier temporaire du disque pour être propre
            os.remove(tmp_path)

# On affiche tous les anciens messages stockés dans l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie : On attend que l'utilisateur tape quelque chose
if user_input := st.chat_input("Posez votre question ici..."):
    
    # 1. On affiche tout de suite le message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. On ajoute ce message à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 3. On vérifie si le cours a bien été chargé avant de répondre
    if "vectorstore" in st.session_state:
        
        with st.chat_message("assistant"):
            with st.spinner("Gemma réfléchit..."):
                
                # --- CONFIGURATION DU RAG ---
                
                # 1. Le Retriever
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

                # 2. Le LLM
                llm = ChatOllama(model="gemma:2b", temperature=0)

                # 3. Le Prompt
                template = """Tu es un assistant pour étudiant ingénieur. 
                Réponds à la question uniquement en te basant sur le contexte suivant.
                Sois concis et précis.
                
                Contexte: {context}
                
                Question: {question}
                
                Réponse:"""
                
                prompt_template = ChatPromptTemplate.from_template(template)
                
                # 4. Parseur de sortie
                output_parser = StrOutputParser()
                
                # 5. La Chaîne Globale : Correction ici
                rag_chain = (
                    {
                        "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                        "question": RunnablePassthrough()
                    }
                    | prompt_template
                    | llm
                    | output_parser
                )
                
                # --- EXÉCUTION ---
                
                # On lance la chaîne avec la question de l'utilisateur
                response = rag_chain.invoke(user_input)
                
                # On affiche la réponse
                st.markdown(response)
                
                # On ajoute la réponse à l'historique pour qu'elle reste affichée
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        # Si aucun fichier n'est chargé
        st.error("Veuillez d'abord charger un cours PDF dans la barre latérale.")