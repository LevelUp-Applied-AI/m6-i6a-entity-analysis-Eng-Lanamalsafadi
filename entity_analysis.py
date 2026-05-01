"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from itertools import combinations
from collections import Counter

def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame unchanged
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None


def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    # TODO: Copy df, apply unicodedata.normalize('NFC', t) to each
    #       text, branch on language for English vs. Arabic handling,
    #       write results into a new `processed_text` column
    df_copy = df.copy()
    
    def normalize_row(row):
        # Apply Unicode NFC normalization
        normalized = unicodedata.normalize('NFC', str(row['text']))
        # Language-aware handling
        if row['language'] == 'en':
            return normalized
        else:
            # For Arabic or others, return normalized text or empty string per instructions
            return normalized 

    df_copy['processed_text'] = df_copy.apply(normalize_row, axis=1)
    return df_copy


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to language == 'en', process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    en_df = df[df['language'] == 'en']
    entity_data = []

    # Using nlp.pipe is more efficient for large batches
    for doc, text_id in zip(nlp.pipe(en_df['text']), en_df['id']):
        for ent in doc.ents:
            entity_data.append({
                'text_id': text_id,
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
            
    return pd.DataFrame(entity_data)


def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count
                           (or filter to co_count >= 2) so the result
                           stays readable on the full corpus.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    # TODO: Count entity frequencies (top 20), compute label totals,
    #       build co-occurrence pairs, and join on articles_df.id to
    #       compute per-category entity-label counts
    # 1. Top 20 entities
    top_entities = entity_df.groupby(['entity_text', 'entity_label']).size().reset_index(name='count')
    top_entities = top_entities.sort_values(by='count', ascending=False).head(20)

    # 2. Label counts
    label_counts = entity_df['entity_label'].value_counts().to_dict()

    # 3. Co-occurrence
    pair_counts = Counter()
    for _, group in entity_df.groupby('text_id'):
        unique_entities = sorted(list(set(group['entity_text'])))
        if len(unique_entities) >= 2:
            for combo in combinations(unique_entities, 2):
                pair_counts[combo] += 1
    
    co_df = pd.DataFrame([
        {'entity_a': p[0], 'entity_b': p[1], 'co_count': c} 
        for p, c in pair_counts.items()
    ])
    co_df = co_df.sort_values(by='co_count', ascending=False).head(50)

    # 4. Per-category patterns
    merged = entity_df.merge(articles_df[['id', 'category']], left_on='text_id', right_on='id')
    per_category = merged.groupby(['category', 'entity_label']).size().reset_index(name='count')

    return {
        'top_entities': top_entities,
        'label_counts': label_counts,
        'co_occurrence': co_df,
        'per_category': per_category
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    # TODO: Create a horizontal bar chart of top entities, colored or
    #       grouped by entity type, save to output_path
    top_df = stats['top_entities']
    
    plt.figure(figsize=(10, 8))
    # Color mapping by label for better visualization
    unique_labels = top_df['entity_label'].unique()
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    
    plt.barh(top_df['entity_text'], top_df['count'], 
             color=[label_color_map[l] for l in top_df['entity_label']])
    
    plt.xlabel('Frequency (Count)')
    plt.title('Top 20 Entities in Climate Corpus')
    plt.gca().invert_yaxis()  # Highest at the top
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=label_color_map[lbl], label=lbl) for lbl in unique_labels]
    plt.legend(handles=legend_elements, title="Entity Type")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    # TODO: Build a formatted report string from the statistics
    top_5 = stats['top_entities'].head(5)
    top_3_co = co_occurrence.head(3)
    
    report = f"""
ENTITY ANALYSIS REPORT
======================
1. Total Counts per Entity Type:
{stats['label_counts']}

2. Top 5 Most Frequent Entities:
{top_5[['entity_text', 'entity_label', 'count']].to_string(index=False)}

3. Top 3 Co-occurring Entity Pairs:
{top_3_co[['entity_a', 'entity_b', 'co_count']].to_string(index=False)}

SUMMARY:
The dataset reveals a high frequency of Geopolitical Entities (GPE) and Organizations (ORG). 
The frequent co-occurrence of policy-related entities suggests a strong focus on 
international agreements and institutional responses to climate change.
"""
    return report


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
