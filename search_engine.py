import nltk
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

nltk.download('punkt')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, stop_words='english', language='english', **kwargs):
        super().__init__(stop_words=stop_words, **kwargs)
        self.language = language  # Simpan bahasa sebagai atribut instance
        self.stemmer = SnowballStemmer(language)

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        stemmer = self.stemmer  # Pastikan stemmer dipanggil dengan benar
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))


def load_documents():
    """ Memuat kumpulan dokumen """
    datas = pd.read_csv('./abcnews-date-text.csv')
    datas = datas['headline_text']
    newlis = list(datas)
    print(newlis)
    return ["aba decides against community broadcasting licence",
            "act fire witnesses must be aware of defamation",
            "a g calls for infrastructure protection summit",
            "air nz staff in aust strike for pay rise",
            "air nz strike to affect australian travellers",
            "ambitious olsson wins triple jump",
            "antic delighted with record breaking barca",
            "aussie qualifier stosur wastes four memphis match",
            "aust addresses un security council over iraq",
            "australia is locked into war timetable opp",
            "australia to contribute 10 million in aid to iraq",
            "barca take record as robson celebrates birthday in",
            "bathhouse plans move ahead",
            "big hopes for launceston cycling championship",
            "big plan to boost paroo water supplies",
            "blizzard buries united states in bills",
            "brigadier dismisses reports troops harassed in",
            "british combat troops arriving daily in kuwait",
            "bryant leads lakers to double overtime win",
            "bushfire victims urged to see centrelink",
            "businesses should prepare for terrorist attacks",
            "calleri avenges final defeat to eliminate massu",
            "call for ethanol blend fuel to go ahead",
            "carews freak goal leaves roma in ruins",
            "cemeteries miss out on funds",
            "code of conduct toughens organ donation regulations",
            "commonwealth bank cuts fixed home loan rates",
            "community urged to help homeless youth",
            "council chief executive fails to secure position",
            "councillor to contest wollongong as independent",
            "council moves to protect tas heritage garden",
            "council welcomes ambulance levy decision",
            "council welcomes insurance breakthrough",
            "crean tells alp leadership critics to shut up",
            "dargo fire threat expected to rise",
            "death toll continues to climb in south korean subway",
            "dems hold plebiscite over iraqi conflict",
            "dent downs philippoussis in tie break thriller",
            "de villiers to learn fate on march 5",
            "digital tv will become commonplace summit",
            "direct anger at govt not soldiers crean urges",
            "dispute over at smithton vegetable processing plant",
            "dog mauls 18 month old toddler in nsw",
            "dying korean subway passengers phoned for help",
            "england change three for wales match",
            "epa still trying to recover chemical clean up costs",
            "expressions of interest sought to build livestock",
            "fed opp to re introduce national insurance",
            "firefighters contain acid spill",
            "four injured in head on highway crash",
            "freedom records net profit for third successive",
            "funds allocated for domestic violence victims",
            "funds allocated for youth at risk",
            "funds announced for bridge work",
            "funds to go to cadell upgrade",
            "funds to help restore cossack",
            "german court to give verdict on sept 11 accused",
            "gilchrist backs rest policy",
            "girl injured in head on highway crash",
            "gold coast to hear about bilby project",
            "golf club feeling smoking ban impact",
            "govt is to blame for ethanols unpopularity opp",
            "greens offer police station alternative",
            "griffiths under fire over project knock back",
            "group to meet in north west wa over rock art",
            "hacker gains access to eight million credit cards",
            "hanson is grossly naive over nsw issues costa",
            "hanson should go back where she came from nsw mp",
            "harrington raring to go after break",
            "health minister backs organ and tissue storage",
            "heavy metal deposits survey nearing end",
            "injured rios pulls out of buenos aires open",
            "inquest finds mans death accidental",
            "investigations underway into death toll of korean",
            "investigation underway into elster creek spill",
            "iraqs neighbours plead for continued un inspections",
            "iraq to pay for own rebuilding white house",
            "irish man arrested over omagh bombing",
            "irrigators vote over river management",
            "israeli forces push into gaza strip",
            "jury to consider verdict in murder case",
            "juvenile sex offenders unlikely to reoffend as",
            "kelly disgusted at alleged bp ethanol scare",
            "kelly not surprised ethanol confidence low",
            "korean subway fire 314 still missing",
            "last minute call hands alinghi big lead",
            "low demand forces air service cuts",
            "man arrested after central qld hijack attempt",
            "man charged over cooma murder",
            "man fined after aboriginal tent embassy raid",
            "man jailed over keno fraud",
            "man with knife hijacks light plane",
            "martin to lobby against losing nt seat in fed",
            "massive drug crop discovered in western nsw",
            "mayor warns landfill protesters",
            "meeting to consider tick clearance costs",
            "meeting to focus on broken hill water woes",
            "moderate lift in wages growth",
            "more than 40 pc of young men drink alcohol at",
            "more water restrictions predicted for northern tas"]


def search_engine(query, documents):
    """ Mencari dokumen yang relevan berdasarkan query dan mengurutkan berdasarkan nilai TF-IDF """
    vectorizer = StemmedTfidfVectorizer(stop_words='english', language='english')
    tfidf_matrix = vectorizer.fit_transform(documents)  # Konversi dokumen ke vektor TF-IDF
    query_vector = vectorizer.transform([query])  # Konversi query ke vektor TF-IDF

    feature_names = vectorizer.get_feature_names_out()
    query_tfidf_values = query_vector.toarray().flatten()

    # Menampilkan daftar kata dengan indeksnya dalam vektor TF-IDF
    print("\nDaftar kata dan indeks dalam vektor TF-IDF:")
    for i, word in enumerate(feature_names):
        print(f"{i}: {word}")

    # Buat dictionary untuk menyimpan nilai TF-IDF dari query
    tfidf_scores = {feature_names[i]: query_tfidf_values[i] for i in range(len(feature_names)) if
                    query_tfidf_values[i] > 0}
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  # Hitung cosine similarity
    distances = euclidean_distances(query_vector, tfidf_matrix).flatten()  # Hitung Euclidean similarity

    ranked_indices = np.argsort(similarities)[::-1]  # Urutkan hasil berdasarkan relevansi

    results = [(documents[i], similarities[i], distances[i]) for i in ranked_indices if similarities[i] > 0]

    # Menampilkan matriks Cosine Similarity
    print("\nMatriks Cosine Similarity:")
    print(cosine_similarity(tfidf_matrix))

    # Menampilkan matriks Euclidean Distance
    print("\nMatriks Euclidean Distance:")
    print(euclidean_distances(tfidf_matrix))

    return results, sorted_tfidf


if __name__ == "__main__":
    documents = load_documents()
    query = input("Masukkan kata kunci pencarian: ")
    results, sorted_tfidf = search_engine(query, documents)

    print("\nNilai TF-IDF untuk kata dalam query:")
    for word, score in sorted_tfidf:
        print(f"{word}: {score:.4f}")

    if results:
        print("\nHasil pencarian berdasarkan relevansi:")
        for idx, (doc, cos_score, euc_dist) in enumerate(results, 1):
            print(f"{idx}. Relevansi (Cosine): {cos_score:.4f}, Jarak (Euclidean): {euc_dist:.4f} - {doc}")
    else:
        print("Tidak ada hasil yang cocok dengan pencarian Anda.")
