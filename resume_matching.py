import math

SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JDS = [
    ("JD-1", "Kakao (ML Engineer)",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver (Backend Engineer)",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis"),
    ("JD-3", "Line (Frontend Engineer)",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"),
]

def normalize_skills(raw):
    raw_lower = raw.lower()
    # Try multi-word matches first
    # Sort aliases by length desc to match longer phrases first
    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=lambda x: len(x), reverse=True)
    
    tokens = [t.strip() for t in raw_lower.split(",")]
    result = []
    for token in tokens:
        matched = False
        # Try multi-word (token as-is after stripping)
        for alias in sorted_aliases:
            if " " in alias and alias in token:
                result.append(SKILL_ALIASES[alias])
                matched = True
                break
        if not matched:
            token_clean = token.strip()
            if token_clean in SKILL_ALIASES:
                result.append(SKILL_ALIASES[token_clean])
    return result

def deduplicate(skills):
    seen = set()
    out = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# Step 1 & 2: Normalize and deduplicate
normalized_resumes = []
for name, raw in RESUMES:
    skills = normalize_skills(raw)
    skills = deduplicate(skills)
    normalized_resumes.append((name, skills))
    print(f"{name}: {skills}")

# Step 3: Build vocabulary
vocab_set = set()
for _, skills in normalized_resumes:
    vocab_set.update(skills)
vocab = sorted(vocab_set)
print(f"\nVocabulary ({len(vocab)}): {vocab}")

# Step 4: Compute TF-IDF
N_docs = len(normalized_resumes)

# df per skill
df = {skill: 0 for skill in vocab}
for _, skills in normalized_resumes:
    for skill in skills:
        df[skill] += 1

# IDF
idf = {skill: math.log(N_docs / df[skill]) for skill in vocab}

# TF-IDF vectors
tfidf_vectors = []
for name, skills in normalized_resumes:
    N = len(skills)
    vec = []
    for skill in vocab:
        if skill in skills:
            tf = 1 / N
            vec.append(tf * idf[skill])
        else:
            vec.append(0.0)
    tfidf_vectors.append((name, vec))

# Step 5: Build JD binary vectors
def normalize_jd_skills(raw):
    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=lambda x: len(x), reverse=True)
    tokens = [t.strip() for t in raw.lower().split(",")]
    result = []
    for token in tokens:
        matched = False
        for alias in sorted_aliases:
            if " " in alias and alias in token:
                result.append(SKILL_ALIASES[alias])
                matched = True
                break
        if not matched:
            token_clean = token.strip()
            if token_clean in SKILL_ALIASES:
                result.append(SKILL_ALIASES[token_clean])
    return list(set(result))

jd_vectors = []
for jd_id, jd_name, jd_raw in JDS:
    jd_skills = normalize_jd_skills(jd_raw)
    vec = [1 if skill in jd_skills else 0 for skill in vocab]
    jd_vectors.append((jd_id, jd_name, vec, jd_skills))
    print(f"\n{jd_id} normalized skills: {jd_skills}")

# Step 6: Cosine similarity
def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

for jd_id, jd_name, jd_vec, _ in jd_vectors:
    scores = []
    for name, resume_vec in tfidf_vectors:
        sim = cosine_sim(resume_vec, jd_vec)
        scores.append((name, sim))
    # Sort by score desc, then name asc for ties
    scores.sort(key=lambda x: (-x[1], x[0]))
    top3 = scores[:3]
    print(f"\n{jd_id} — {jd_name}")
    result_str = ", ".join(f"{name}({score:.2f})" for name, score in top3)
    print(result_str)

