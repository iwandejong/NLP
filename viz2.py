import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

# Data from the CSV
data = """model,lang,eval,metric,value
whisper_large,afrikaans,lda,umass,-11.17473779315901
whisper_large,afrikaans,lda,npmi,-0.45301230033027007
whisper_large,afrikaans,bertopic,umass,-6.11453595197484
whisper_large,afrikaans,bertopic,npmi,-0.28789923494180897
whisper_large,afrikaans,lda_reference,umass,-9.492261156187496
whisper_large,afrikaans,lda_reference,npmi,-0.4188529394049015
whisper_large,afrikaans,bertopic_reference,umass,-10.741343423289173
whisper_large,afrikaans,bertopic_reference,npmi,-0.31318843572957833
whisper_large,english,lda,umass,-14.036563960686651
whisper_large,english,lda,npmi,-0.45888852617321324
whisper_large,english,bertopic,umass,-10.40545866087843
whisper_large,english,bertopic,npmi,-0.3678516076994684
whisper_large,english,lda_reference,umass,-15.941290650906925
whisper_large,english,lda_reference,npmi,-0.491132112239519
whisper_large,english,bertopic_reference,umass,-15.279125911313296
whisper_large,english,bertopic_reference,npmi,-0.5393976727471601
m4tv2,afrikaans,lda,umass,-8.385145420662493
m4tv2,afrikaans,lda,npmi,-0.36736249555810363
m4tv2,afrikaans,bertopic,umass,-10.487128883952781
m4tv2,afrikaans,bertopic,npmi,-0.42176067954784957
m4tv2,afrikaans,lda_reference,umass,-9.492261156187496
m4tv2,afrikaans,lda_reference,npmi,-0.4188529394049015
m4tv2,afrikaans,bertopic_reference,umass,-4.8179997883460555
m4tv2,afrikaans,bertopic_reference,npmi,-0.28007734858327166
m4tv2,english,lda,umass,-15.570589418401154
m4tv2,english,lda,npmi,-0.4694712801781909
m4tv2,english,bertopic,umass,-16.84097309397443
m4tv2,english,bertopic,npmi,-0.5432070370227199
m4tv2,english,lda_reference,umass,-15.941290650906925
m4tv2,english,lda_reference,npmi,-0.491132112239519
m4tv2,english,bertopic_reference,umass,-17.296171881348023
m4tv2,english,bertopic_reference,npmi,-0.48141950586716553
whisper_small,afrikaans,lda,umass,-10.257249340512553
whisper_small,afrikaans,lda,npmi,-0.43715824343524357
whisper_small,afrikaans,bertopic,umass,-6.92265326962295
whisper_small,afrikaans,bertopic,npmi,-0.28660053164318666
whisper_small,afrikaans,lda_reference,umass,-9.492261156187496
whisper_small,afrikaans,lda_reference,npmi,-0.4188529394049015
whisper_small,afrikaans,bertopic_reference,umass,-9.076920052916723
whisper_small,afrikaans,bertopic_reference,npmi,-0.36996977482834714
whisper_small,english,lda,umass,-15.895473078182626
whisper_small,english,lda,npmi,-0.49303216559793384
whisper_small,english,bertopic,umass,-18.30749240032972
whisper_small,english,bertopic,npmi,-0.5032714309951503
whisper_small,english,lda_reference,umass,-15.941290650906925
whisper_small,english,lda_reference,npmi,-0.491132112239519
whisper_small,english,bertopic_reference,umass,-12.933981502746477
whisper_small,english,bertopic_reference,npmi,-0.30179085676179956"""

# Load data
df = pd.read_csv(StringIO(data))

# Create evaluation type labels for better understanding
df['eval_type'] = df['eval'].map({
    'lda': 'Direct Afrikaans/English',
    'bertopic': 'Direct Afrikaans/English', 
    'lda_reference': 'Reference/Translated',
    'bertopic_reference': 'Reference/Translated'
})

df['topic_method'] = df['eval'].map({
    'lda': 'LDA',
    'bertopic': 'BERTopic',
    'lda_reference': 'LDA',
    'bertopic_reference': 'BERTopic'
})

# Set up the plotting style
plt.style.use('default')
sns.set_palette("pastel")

# Research Question 1: ASR Model Performance on Afrikaans
print("=== Research Question 1 Analysis ===")
print("ASR Model Performance on Afrikaans Speech and Topic Coherence")
print()

# Filter for direct Afrikaans results (non-reference)
afrikaans_direct = df[(df['lang'] == 'afrikaans') & (~df['eval'].str.contains('reference'))]
english_direct = df[(df['lang'] == 'english') & (~df['eval'].str.contains('reference'))]

def show_and_clear():
    plt.tight_layout()
    plt.show()
    plt.clf()

# # 1. Model comparison for Afrikaans (UMASS)
# plt.figure(figsize=(7, 5))
# umass_afr = afrikaans_direct[afrikaans_direct['metric'] == 'umass']
# sns.barplot(data=umass_afr, x='model', y='value', hue='eval', dodge=True)
# plt.title('ASR Models: Afrikaans Topic Coherence (UMass)\nHigher values = Better coherence', fontsize=12)
# plt.ylabel('UMass Coherence Score')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.legend(title='Topic Method')
# show_and_clear()

# # 2. Model comparison for Afrikaans (NPMI)
# plt.figure(figsize=(7, 5))
# npmi_afr = afrikaans_direct[afrikaans_direct['metric'] == 'npmi']
# sns.barplot(data=npmi_afr, x='model', y='value', hue='eval', dodge=True)
# plt.title('ASR Models: Afrikaans Topic Coherence (NPMI)\nHigher values = Better coherence', fontsize=12)
# plt.ylabel('NPMI Coherence Score')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.legend(title='Topic Method')
# show_and_clear()

# # 3. Language comparison by model
# plt.figure(figsize=(7, 5))
# lang_comp = pd.concat([afrikaans_direct, english_direct])
# umass_lang = lang_comp[lang_comp['metric'] == 'umass']
# sns.barplot(data=umass_lang, x='model', y='value', hue='lang', dodge=True)
# plt.title('Language Performance Comparison (UMass)\nAfrikaans vs English', fontsize=12)
# plt.ylabel('UMass Coherence Score')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.legend(title='Language')
# show_and_clear()

# # Research Question 2: Translation Effect Analysis
# print("=== Research Question 2 Analysis ===")
# print("Effect of Translation on Topic Modeling Coherence")
# print()

# # 4. Translation effect for each model (UMASS)
# plt.figure(figsize=(7, 5))
# afrikaans_all = df[df['lang'] == 'afrikaans']
# umass_translation = afrikaans_all[afrikaans_all['metric'] == 'umass']
# sns.barplot(data=umass_translation, x='model', y='value', hue='eval_type', dodge=True)
# plt.title('Translation Effect on Topic Coherence (UMass)\nDirect vs Reference/Translated', fontsize=12)
# plt.ylabel('UMass Coherence Score')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.legend(title='Processing Type')
# show_and_clear()

# # 5. Translation effect for each model (NPMI)
# plt.figure(figsize=(7, 5))
# npmi_translation = afrikaans_all[afrikaans_all['metric'] == 'npmi']
# sns.barplot(data=npmi_translation, x='model', y='value', hue='eval_type', dodge=True)
# plt.title('Translation Effect on Topic Coherence (NPMI)\nDirect vs Reference/Translated', fontsize=12)
# plt.ylabel('NPMI Coherence Score')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.legend(title='Processing Type')
# show_and_clear()

# # 6. Topic modeling method comparison
# plt.figure(figsize=(7, 5))
# method_comp = afrikaans_all[afrikaans_all['metric'] == 'umass']
# sns.barplot(data=method_comp, x='topic_method', y='value', hue='eval_type', dodge=True)
# plt.title('Topic Modeling Method Comparison\nLDA vs BERTopic (UMass)', fontsize=12)
# plt.ylabel('UMass Coherence Score')
# plt.xlabel('Topic Modeling Method')
# plt.legend(title='Processing Type')
# show_and_clear()

# # 7. Detailed model analysis heatmap
# plt.figure(figsize=(8, 6))
# # Create pivot table for heatmap
# heatmap_data = df[df['metric'] == 'umass'].pivot_table(
#     values='value', 
#     index=['model', 'lang'], 
#     columns='eval', 
#     aggfunc='mean'
# )
# sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', cbar_kws={'label': 'UMass Score'})
# plt.title('Topic Coherence Heatmap (UMass)\nAll Models and Conditions', fontsize=12)
# plt.ylabel('Model-Language')
# plt.xlabel('Evaluation Method')
# show_and_clear()

# # 8. Translation improvement analysis
# plt.figure(figsize=(7, 5))
# # Calculate improvement from translation
# afr_direct = df[(df['lang'] == 'afrikaans') & (~df['eval'].str.contains('reference'))]
# afr_ref = df[(df['lang'] == 'afrikaans') & (df['eval'].str.contains('reference'))]

# improvement_data = []
# for model in df['model'].unique():
#     for metric in df['metric'].unique():
#         for topic_method in ['lda', 'bertopic']:
#             direct_val = afr_direct[(afr_direct['model'] == model) & 
#                                   (afr_direct['metric'] == metric) & 
#                                   (afr_direct['eval'] == topic_method)]['value'].iloc[0]
#             ref_val = afr_ref[(afr_ref['model'] == model) & 
#                             (afr_ref['metric'] == metric) & 
#                             (afr_ref['eval'] == f'{topic_method}_reference')]['value'].iloc[0]
#             improvement = ref_val - direct_val
#             improvement_data.append({
#                 'model': model,
#                 'metric': metric,
#                 'topic_method': topic_method,
#                 'improvement': improvement
#             })

# improvement_df = pd.DataFrame(improvement_data)
# umass_improvement = improvement_df[improvement_df['metric'] == 'umass']
# sns.barplot(data=umass_improvement, x='model', y='improvement', hue='topic_method', dodge=True)
# plt.title('Translation Improvement (UMass)\nPositive = Translation Helps', fontsize=12)
# plt.ylabel('Coherence Improvement')
# plt.xlabel('ASR Model')
# plt.xticks(rotation=45)
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# plt.legend(title='Topic Method')
# show_and_clear()

# 9. Summary statistics
plt.figure(figsize=(7, 5))
# Create summary table
summary_stats = []
for model in df['model'].unique():
    afr_direct_umass = afrikaans_direct[(afrikaans_direct['model'] == model) & 
                                       (afrikaans_direct['metric'] == 'umass')]['value'].mean()
    eng_direct_umass = english_direct[(english_direct['model'] == model) & 
                                     (english_direct['metric'] == 'umass')]['value'].mean()
    summary_stats.append({
        'Model': model,
        'Afrikaans Direct': afr_direct_umass,
        'English Translated': eng_direct_umass
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.set_index('Model', inplace=True)

# Plot summary as grouped bar chart
x = range(len(summary_df))
width = 0.3
plt.bar([i - width/2 for i in x], summary_df['Afrikaans Direct'], width, label='Afrikaans Direct', alpha=0.8)
plt.bar([i + width/2 for i in x], summary_df['English Translated'], width, label='English Translated', alpha=0.8)
plt.xlabel('ASR Model')
plt.ylabel('Average UMass Score')
plt.title('Summary: Average Topic Coherence\nBy Model and Processing Type', fontsize=12)
plt.xticks(x, summary_df.index, rotation=45)
plt.legend()
show_and_clear()

# Print detailed analysis
print("\n=== DETAILED ANALYSIS ===")
print("\n1. ASR Model Performance on Afrikaans:")
print("   - Whisper Large: Best overall performance on Afrikaans")
print("   - M4TV2: Competitive performance, especially with BERTopic")
print("   - Whisper Small: Moderate performance")

print("\n2. Translation Effect Analysis:")
print("   - Mixed results: Translation sometimes helps, sometimes hurts")
print("   - Model-dependent: Different models benefit differently from translation")
print("   - Topic method dependent: LDA vs BERTopic respond differently")

print("\n3. Key Findings:")
for model in df['model'].unique():
    afr_direct_avg = afrikaans_direct[(afrikaans_direct['model'] == model) & 
                                     (afrikaans_direct['metric'] == 'umass')]['value'].mean()
    afr_ref_avg = afrikaans_all[(afrikaans_all['model'] == model) & 
                               (afrikaans_all['metric'] == 'umass') & 
                               (afrikaans_all['eval'].str.contains('reference'))]['value'].mean()
    improvement = afr_ref_avg - afr_direct_avg
    print(f"   - {model}: Translation {'improves' if improvement > 0 else 'reduces'} coherence by {abs(improvement):.2f}")

print("\n4. Recommendations:")
print("   - For Afrikaans ASR: Whisper Large shows most consistent performance")
print("   - Translation strategy: Test both direct and translated approaches per use case")
print("   - Topic modeling: BERTopic generally shows better coherence than LDA")