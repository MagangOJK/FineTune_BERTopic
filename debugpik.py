import streamlit as st
import pandas as pd
import re
import io
import json
import torch
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords as nltk_stopwords
import nltk
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

try:
    if hasattr(torch, "_classes") and hasattr(torch._classes, "__path__"):
        torch._classes.__path__ = []
except:
    pass

nltk.download('stopwords')

common_terms = {
    "laporan", "apolo", "pelaporan", "kami", "saya", "yang", "fraud", "strategi", "aplikasi", "dan", "ojk","untuk","pada","as","di",
    "anti", "kendala", "bapak", "ibu", "ini", "itu", "pengkinian", "perusahaan", "surat", "penerapan", "no",
    "go", "fff", "id", "izin", "sttd", "helpdesk", "lestari", "april", "deadline", "gita", "trimegah", "pt"
}

st.set_page_config(page_title="🔍 Ekstraksi Topik • BERTopic", layout="wide", page_icon="📊")
st.markdown("# 📊 Ekstraksi Topik dengan BERTopic 🔍")

st.sidebar.markdown("## 🗂️ Upload File")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV/Excel", type=['csv','xlsx'])
if not uploaded_file:
    st.info("➡️ Silakan upload file terlebih dahulu.")
    st.stop()
try:
    if uploaded_file.name.endswith('xlsx'):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Pilih Sheet 📄", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.sidebar.error(f"❌ Error membaca file: {e}")
    st.stop()


st.subheader("👀 Preview Data")
st.dataframe(df.head())
col = st.selectbox("📝 Kolom teks kendala terdapat pada kolom:", df.columns)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Opsi Preprocessing")
use_stemming = st.sidebar.checkbox("✨ Lakukan Stemming (Sastrawi + cache)", value=True)
user_remove_txt = st.sidebar.text_area("🗑️ Kata tidak penting (pisah koma)", "")
if user_remove_txt.strip():
    user_terms = {w.strip().lower() for w in user_remove_txt.split(",") if w.strip()}
    remove_list = common_terms.union(user_terms)
else:
    remove_list = common_terms

@st.cache_data(show_spinner=False)
def load_stem_cache(path="E:\\DOWNLOAD\\stem_cache.json"):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}
    
@st.cache_data(show_spinner=False)
def get_stemmed_mapping(text_list):
    try:
        with open("E:\\DOWNLOAD\\stem_cache.json", 'r') as f:
            static_cache = json.load(f)
    except:
        static_cache = {}
    stemmer = StemmerFactory().create_stemmer()
    stop_factory = StopWordRemoverFactory()
    combined_stopwords = set(stop_factory.get_stop_words() + nltk_stopwords.words('english')) | common_terms | set(remove_list)
    runtime_cache = {}

    def clean_and_stem(t):
        t = str(t).lower()
        # Defining substitutions
        subs = {
            r"\bmasuk\b": "login",
            r"\blog-in\b": "login",
            r"\brenbis\b": "rencana bisnis",
            r"\baro\b": "administrator responsible officer",
            r"\bro\b": "responsible officer",
            r"\bmengunduh\b": "download",
            r"\bunduh\b": "download"
        }
        for pat, rep in subs.items():
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        t = re.sub(r'[^a-z\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        words = [w for w in t.split() if w not in combined_stopwords]
        stemmed_words = []
        for w in words:
            if w in static_cache:
                stemmed_words.append(static_cache[w])
            elif w in runtime_cache:
                stemmed_words.append(runtime_cache[w])
            else:
                stemmed = stemmer.stem(w)
                runtime_cache[w] = stemmed
                stemmed_words.append(stemmed)
        return ' '.join(stemmed_words)

    return {text: clean_and_stem(text) for text in text_list}

# Fungsi pre-cleaning sebelum BERTopic
@st.cache_data(show_spinner=False)
def preprocess_text(text):
    def remove_dear_ojk(t):
        return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", t, flags=re.IGNORECASE) if isinstance(t, str) else t
    def extract_complaint(t):
        if not isinstance(t, str):
            return ""
        start_p = [r"PERHATIAN: E-mail ini berasal dari pihak di luar OJK.*?attachment.*?link.*?yang terdapat pada e-mail ini."]
        end_p = [
            r"(From\s*.*?From|Best regards|Salam|Atas perhatiannya|Regards|Best Regards|Mohon\s*untuk\s*melengkapi\s*data\s*.*tabel\s*dibawah).*,?",
            r"From:\s*Direktorat\s*Pelaporan\s*Data.*"
        ]
        m_start = None
        for pat in start_p:
            matches = list(re.finditer(pat, t, re.DOTALL|re.IGNORECASE))
            if matches:
                m_start = matches[-1].end()
        if m_start:
            t = t[m_start:].strip()
        for pat in end_p:
            m = re.search(pat, t, re.DOTALL|re.IGNORECASE)
            if m:
                t = t[:m.start()].strip()
        sens = [
            r"Nama\s*Terdaftar\s*.*",
            r"Email\s*.*",
            r"No\.\s*Telp\s*.*",
            r"User\s*Id\s*/\s*User\s*Name\s*.*",
            r"No\.\s*KTP\s*.*",
            r"Nama\s*Perusahaan\s*.*",
            r"Nama\s*Pelapor\s*.*",
            r"No\.\s*Telp\s*Pelapor\s*.*",
            r"Internal",
            r"Dengan\s*hormat.*",
            r"Jenis\s*Usaha\s*.*",
            r"Keterangan\s*.*",
            r"No\.\s*SK\s*.*",
            r"Alamat\s*website/URL\s*.*",
            r"Selamat\s*(Pagi|Siang|Sore).*",
            r"Kepada\s*Yth\.\s*Bapak/Ibu.*",
            r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*",
            r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*",
            r"No\.\s*NPWP\s*Perusahaan\s*.*",
            r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*",
            r"Yth\s*.*",
            r"demikian\s*.*",
            r"Demikian\s*.*",
            r"Demikianlah\s*.*"
        ]
        for pat in sens:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        return t
    def clean_email_text(t):
        pats = [
            r"(?i)(terlampir|mohon\s*bantuan|terima\s*kasi|yth|daripada\s*\w+\s*\@.*?\.com)",
            r"(?i)Selamat\s*.*",
            r"(?i)Dear\s*(Bapak/Ibu\s*)?Helpdesk\s*OJK",
            r"(?i)Dear\s*Helpdesk",
            r"(?i)Mohon\s*bantuannya",
            r"(?i)Terimakasih",
            r"(?i)Hormat\s*kami",
            r"(?i)Regard",
            r"(?i)Atas\s*perhatian\s*dan\s*kerja\s*samanya",
            r"(?i)Selamat\s*pagi",
            r"(?i)Dengan\s*hormat",
            r"(?i)Perhatian",
            r"(?i)Caution",
            r"(?i)Peringatan",
            r"(?i)Harap\s*diperhatikan",
            r"(?i)Terlampir",
            r"(?i)Mohon\s*kerjasamanya",
            r"(?i)Mohon\s*informasi",
            r"(?i)Sehubungan\s*dengan",
            r"(?i)Kepada",
            r"(?i)Kami\s*moho",
            r"(?i)Terkait",
            r"(?i)Berikut\s*kami\s*sampaikan",
            r"(?i)Jika\s*Anda\s*bukan\s*penerima\s*yang\s*dimaksud",
            r"(?i)Email\s*ini\s*hanya\s*ditujukan\s*untuk\s*penerima",
            r"(?i)Mohon\s*segera\s*memberitahukan\s*kami",
            r"(?i)Jika\s*Anda\s*memerima\s*ini\s*secara\s*tidak\s*seganja",
            r"(?i)Mohon\s*dihapus",
            r"(?i)Kami\s*mengucapkan\s*terima\s*kasi",
            r"(?i)Mohon\s*perhatian",
            r"(?i)From\s*.*",
            r"(?i)\n+",
            r"(?i)\s{2,}",
            r"(?i):\s*e-Mail\s*ini\s*termasuk\s∗seluruh\s∗lampirannya\s∗,\s∗bilang\s∗adatermasuk\s*seluruh\s*lampirannya\s*,\s*bilang\s*ada\s*hanya\s*ditujukan\s*penerima\s*yang\s*tercantum\s*di\s*atas.*",
            r"(?i):\s*This\s*electronic\s*mail\s*and\s*/\s*or\s*any\s*files\s*transmitted\s*with\s*it\s*may\s*contain\s*confidential\s*or\s*copyright\s*PT\.\s*Jasa\s*Raharja.*",
            r"(?i)PT\.\s*Jasa\s*Raharja\s*tidak\s*bertanggung\s*jawab\s*atas\s*kerugian\s*yang\s*ditimbulkan\s*oleh\s*virus\s*yang\s*ditularkan\s*melalui\s*e-Mail\s*ini.*",
            r"(?i)Jika\s*Anda\s*secara\s*tidak\s*seganja\s*menerima\s*e-Mail\s*ini\s*,\s*untuk\s*segera\s*memberitahukan\s*ke\s*alamat\s*e-Mail\s*pengirim\s*serta\s*menghapus\s*e-Mail\s*ini\s*beserta\s*seluruh\s*lampirannya\s*.*",
            r"(?i)\s*Please\s*reply\s*to\s*this\s*electronic\s*mail\s*to\s*notify\s*the\s*sender\s*of\s*its\s*incorrect\s*delivery\s*,\s*and\s*then\s*delete\s*both\s*it\s*and\s*your\s*reply.*",
            ]
        for pat in pats:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        return t
    def cut_off_general(c):
        cut_keywords = [
           "PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo",
            "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan",
            "h._________",
            "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus",
            "KamiBapak/Ibu untuk pencerahannya",
            "kami ucapkan h. ,",
            "-- , DANA PENSIUN BPD JAWA",
            "sDian PENYANGKALAN.",
            "------------------------Dari: Adrian",
            "hormat saya RidwanForwarded",
            "--h, DANA PENSIUN WIJAYA",
            "Mohon InfonyahKantor",
            "an arahannya dari Bapak/ Ibu",
            "ya untuk di check ya.Thank",
            "Kendala:Thank youAddelin",
            ",Sekretaris DAPENUrusan Humas & ProtokolTazkya",
            "Mohon arahannya.Berikut screenshot",
            "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024",
            "Annie Clara DesiantyComplianceIndonesia",
            "Dian Rosmawati RambeCompliance",
            "Beararti apakah,Tri WahyuniCompliance DeptPT.",
            "Dengan alamat email yang didaftarkan",
            "dan arahan",
            ",AJB Bumiputera",
            "’h sebelumnya Afriyanty",
            "PENYANGKALAN.",
            "h Dana Pensiun PKT",
            ", h , Tasya PT.",
            "Contoh: 10.00",
            "hAnnisa Adelya SerawaiPT Fazz",
            "sebagaimana gambar di bawah ini",
            "PT Asuransi Jiwa SeaInsure On Fri",
            "hJana MaesiatiBanking ReportFinance",
            "Tembusan",
            "Sebagai referensi",
            "hAdriansyah",
            "h atas bantuannya Dwi Anggina",
            "PT Asuransi Jiwa SeaInsure",
            "dengan notifikasi dibawah ini",
            "Terima ksh",
            ": DISCLAIMER",
            "Sebagai informasi",
            "nya. h.Kind s,Melati",
            ": DISCLAIMER",
            "Petugas AROPT",
            "h,Julianto",
            "h,Hernawati",
            "Dana Pensiun Syariah",
            ",Tria NoviatyStrategic"
        ]
        for kw in cut_keywords:
            if kw in c:
                c = c.split(kw)[0]
        return c
    t = text if isinstance(text, str) else ""
    t = remove_dear_ojk(t)
    comp = extract_complaint(t)
    comp = clean_email_text(comp)
    comp = cut_off_general(comp)
    subs = {
        r"\bmasuk\b": "login",
        r"\blog-in\b": "login",
        r"\brenbis\b": "rencana bisnis",
        r"\baro\b": "administrator responsible officer",
        r"\bro\b": "responsible officer",
        r"\bmengunduh\b": "download",
        r"\bunduh\b": "download"
    }
    for pat, rep in subs.items():
        comp = re.sub(pat, rep, comp, flags=re.IGNORECASE)
    comp = re.sub(r'[^a-z\s]', ' ', comp.lower())
    comp = re.sub(r'\s+', ' ', comp).strip()
    return comp

st.sidebar.markdown("---")
st.sidebar.markdown("## 📚 Opsi BERTopic & Embedding")
language_choice = st.sidebar.selectbox("🌐 Bahasa BERTopic", ["indonesian", "english", "multilingual"])
embed_option = st.sidebar.selectbox(
    "🤖 Pilih model embedding",
    (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased",
        "indobenchmark/indobert-base-p1",
        "indobenchmark/indobert-base-p2",
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    )
)
calculate_prob = st.sidebar.checkbox("📈 buat probabilitas", value=True)
verbose = st.sidebar.checkbox("💬 verbose BERTopic", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 UMAP Settings")
n_neighbors = st.sidebar.slider("n_neighbors", 5, 50, 10)
n_components = st.sidebar.slider("n_components", 2, 10, 3)
min_dist = st.sidebar.slider("min_dist", 0.0, 0.99, 0.1, step=0.05)
umap_metric = st.sidebar.selectbox("metric", ["cosine", "euclidean", "manhattan", "correlation"])
random_state = st.sidebar.number_input("random_state UMAP", min_value=0, value=1337, step=1)

# HDBSCAN parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗂️ HDBSCAN Settings")
min_cluster_size = st.sidebar.slider("min_cluster_size", 2, 50, 10)
min_samples = st.sidebar.slider("min_samples (0=auto)", 0, 50, 2)
hdbscan_metric = st.sidebar.selectbox("metric", ["euclidean", "manhattan", "cosine", "l1", "l2"])
cluster_selection_method = st.sidebar.selectbox("cluster_selection_method", ["eom", "leaf"])

if st.button("Ekstraksi Topik"):
    with st.spinner("Preprocessing data..."):
        df_filtered = df[df[col].notna()].copy()
        texts_raw = df_filtered[col].astype(str).tolist()
        pre_cleaned = [preprocess_text(txt) for txt in texts_raw]
        if use_stemming:
            unique_texts = list(set(pre_cleaned))
            stem_map = get_stemmed_mapping(unique_texts)
            processed_texts = [stem_map.get(txt, "") for txt in pre_cleaned]
        else:
            processed_texts = pre_cleaned
        if remove_list:
            processed_texts = [" ".join([w for w in txt.split() if w not in remove_list]) for txt in processed_texts]
        valid_indices = [i for i, txt in enumerate(processed_texts) if txt.strip()]
        if not valid_indices:
            st.error("Semua dokumen kosong setelah preprocessing.")
            st.stop()
        texts_for_topic = [processed_texts[i] for i in valid_indices]
    
    with st.spinner("Memuat model embedding..."):
        try:
            embedding_model = SentenceTransformer(embed_option)
        except Exception as e:
            st.error(f"Gagal memuat model embedding: {e}")
            st.stop()

    with st.spinner("Menjalankan BERTopic..."):
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=umap_metric, random_state=int(random_state))
        hdbscan_model = HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples) or None, metric=hdbscan_metric, cluster_selection_method=cluster_selection_method, prediction_data=True)
        topic_model = BERTopic(language=language_choice if language_choice != "multilingual" else None, embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=calculate_prob, verbose=verbose)
        topics, probs = topic_model.fit_transform(texts_for_topic)

    st.subheader("Topic Info")
    topic_info = topic_model.get_topic_info()
    st.dataframe(topic_info)
    
    st.subheader("Visualisasi Topik")
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig, use_container_width=True)

    df_result = df_filtered.copy().reset_index(drop=True)
    full_topics = [-1] * len(df_result)
    for idx, t in zip(valid_indices, topics):
        full_topics[idx] = t
    df_result["Topic"] = full_topics
    df_result["Topic_Label"] = df_result["Topic"].apply(
        lambda tid: "Outlier" if tid == -1 else f"{tid}: " + ", ".join([w for w, _ in topic_model.get_topic(tid)][:3])
    )

    # Topic info sheet
    info_rows = []
    for row in topic_info.itertuples():
        tw = topic_model.get_topic(row.Topic)
        info_rows.append({
            "Topic": row.Topic,
            "Count": row.Count,
            "Top_Words": ", ".join([w for w, _ in tw]) if tw else ""
        })
    df_topic_info = pd.DataFrame(info_rows)

    def to_excel_two_sheets(df_docs, df_topics):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_docs.to_excel(writer, index=False, sheet_name='Hasil_Topik')
            df_topics.to_excel(writer, index=False, sheet_name='Topic_Info')
            for sheet_name, df_ in [("Hasil_Topik", df_docs), ("Topic_Info", df_topics)]:
                worksheet = writer.sheets[sheet_name]
                for idx, col_name in enumerate(df_.columns):
                    series = df_[col_name].astype(str)
                    max_len = min(max(series.map(len).max(), len(col_name)) + 2, 30)
                    worksheet.set_column(idx, idx, max_len)
        return output.getvalue()

    excel_data = to_excel_two_sheets(df_result, df_topic_info)
    st.download_button(
        label="Download .xlsx",
        data=excel_data,
        file_name='hasil_topik_BERTopic.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.markdown("<div style='text-align:center;color:gray;'>Developed by Mesakh Besta Anugrah • OJK Internship 2025</div>", unsafe_allow_html=True)