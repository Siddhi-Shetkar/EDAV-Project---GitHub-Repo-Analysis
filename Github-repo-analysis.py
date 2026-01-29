import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Load CSV (keep in same folder)
# ================================
df = pd.read_csv("repositories.csv")

# Preview data
print(df.head())

# Dataset info
print(df.info())
print("\nMissing values:\n")
print(df.isnull().sum())

# ================================
# Handle column name differences
# ================================
# Standardize star column name
if 'stargazers_count' in df.columns:
    df.rename(columns={'stargazers_count': 'Stars'}, inplace=True)

# Drop rows with missing critical values
df = df.dropna(subset=['Language', 'Stars'])

# ================================
# Date handling
# ================================
if 'Created At' in df.columns:
    df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
    df['year_created'] = df['Created At'].dt.year

if 'Updated At' in df.columns:
    df['Updated At'] = pd.to_datetime(df['Updated At'], errors='coerce')

print("\nStatistical Summary:\n")
print(df.describe())

# ================================
# Top Languages
# ================================
print("\nTop 10 Languages:\n")
top_langs = df['Language'].value_counts().head(10)
print(top_langs)

plt.figure(figsize=(10, 5))
top_langs.plot(kind='bar')
plt.title('Top 10 Programming Languages on GitHub')
plt.xlabel('Language')
plt.ylabel('Number of Repositories')
plt.tight_layout()
plt.show()

# ================================
# Correlation / Pair Plot
# ================================
possible_cols = ['Stars', 'forks_count', 'open_issues_count']
existing_cols = [c for c in possible_cols if c in df.columns]

if len(existing_cols) >= 2:
    sns.pairplot(df[existing_cols].select_dtypes(include='number'))
else:
    print("Not enough numeric columns for correlation plot.")

# ================================
# Language Trends Over Time
# ================================
if 'year_created' in df.columns:
    lang_trend = (
        df.groupby(['year_created', 'Language'])
        .size()
        .reset_index(name='count')
    )

    top5_langs = df['Language'].value_counts().head(5).index

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=lang_trend[lang_trend['Language'].isin(top5_langs)],
        x='year_created',
        y='count',
        hue='Language'
    )
    plt.title('Language Popularity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Repository Count')
    plt.tight_layout()
    plt.show()

# ================================
# Stars Distribution
# ================================
sns.histplot(df[df['Stars'] > 0]['Stars'], bins=50, log_scale=True)
plt.title('Distribution of Repository Stars')
plt.xlabel('Stars')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# ================================
# Top Repositories
# ================================
top_repos = (
    df[['Name', 'Language', 'Stars']]
    .sort_values(by='Stars', ascending=False)
    .head(10)
)

print("\nTop 10 Most Starred Repositories:\n")
print(top_repos)

# ================================
# Repositories Created Per Year
# ================================
if 'year_created' in df.columns:
    year_counts = df['year_created'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    year_counts.plot(kind='line', marker='o')
    plt.title('Repositories Created Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Repositories')
    plt.tight_layout()
    plt.show()
