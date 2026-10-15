# This script loads, cleans and runs analyses on the survey data from the climate stripes dataset.

import pandas as pd
import code
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
import numpy as np
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity

# Load the data.
sheets = ['UK Data_Cleaned','US_Data_Cleaned']

# Read both sheets in one call so the workbook is only opened/parsed once, not once per sheet.
sheet_dfs = pd.read_excel('',engine='openpyxl',sheet_name=sheets)

dfs = []
for sheet in sheets:
	# Cleaning. Row 0 is junk from the formatting of the excel.
	dfs.append(sheet_dfs[sheet][1:])

# Merge to one df (need to review if this makes sense)
df = pd.concat(dfs,ignore_index=True)

# Column descriptions:
# Stripes_Seen - Have you seen this image before? Yes(1)/No(2)/Unsure(3)
# Stripes_Three-Words - In up to three words, how does this image make you feel? Open text
# Stripes_Shows - What do you think this image shows? Please answer in as much detail as you would like: Open text
# Sseen_SocialMedia - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "Social media (e.g. facebook, twitter/x, linkedin, instagram)") 1 or empty
# Sseen_Museum - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "At a museum or gallery") 1 or empty
# Sseen_Community - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "At a community event") 1 or empty
# Sseen_Business - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "At a business event") 1 or empty
# Sseen_Sport - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "At a sporting event (e.g. cycling, football, formula E)") 1 or empty
# Sseen_Music - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "At a music event") 1 or empty
# Sseen_News - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On the news (e.g. on tv, newspapers or online)") 1 or empty
# Sseen_Weather - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On a weather forecast") 1 or empty
# Sseen_Building - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On a building (e.g. projected on as a lightshow)") 1 or empty
# Sseen_Clothing - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On clothing (e.g. at a fashion event, on clothing, or accessories)") 1 or empty
# Sseen_Transport - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On transport (e.g. on a car or bus)") 1 or empty
# Sseen_Book - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "On or in a book or report ") 1 or empty
# Sseen_Politics - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "In politics (e.g. on campaign materials, or at a protest)") 1 or empty
# Sseen_Education - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "In an educational setting (e.g. at school, university or an educational event)") 1 or empty
# Sseen_Other - Here are some places you may have seen the stripes before, please select all that apply to you: (selected "Other (please describe below)") Open text
# Sseen_Details - We'd like to know more about how you saw the image and how it was used. Please tell us more, and give as much detail are you are able to in the text box below. If you have seen this image in multiple places, please choose the example you recall the best. You may wish to tell us about the location of the image, the way it was used, the people/organisations who used it, or any other details you think are relevant. Open text

# Respondents are now shown a description of what the stripes are.

# Sshows_PostPrompt - In your own words, what does this image show? Open text
# Emo_Anxious - How far do you feel that this image makes you feel: (Anxious scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Hopeful - How far do you feel that this image makes you feel: (Hopeful scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Inspired - How far do you feel that this image makes you feel: (Inspired scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Happy - How far do you feel that this image makes you feel: (Happy scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Angry - How far do you feel that this image makes you feel: (Angry scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Sad - How far do you feel that this image makes you feel: (Sad scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Guilty - How far do you feel that this image makes you feel: (Guilty scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Hopeless - How far do you feel that this image makes you feel: (Hopeless scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# Emo_Informed - How far do you feel that this image makes you feel: (Informed scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 

# Image_UnderstandMean - How far do you agree with (I understand what the image means) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_UnderstandCC - How far do you agree with (The image helps me to understand climate change/global warming) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_ActCC - How far do you agree with (The image inspires me to act on climate change/ global warming) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_NoUnderstand - How far do you agree with (I do not understand what the image means) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_Attention - How far do you agree with (This image caught my attention) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_Appeal - How far do you agree with (The image looks visually appealing) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_WouldShare - How far do you agree with (I would share the image myself in the future ) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# Image_LikeColour - How far do you agree with (I like the colours of the image) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree

# Image_WhyAct - You said the image inspired you to act on climate change, can you tell us more about how and why this was? Open text

# Particpants are shown another stripes image of the generational stripes.
# GStripes_Seen - Have you seen this image before? Yes(1)/No(2)/Unsure(3)
# Gstripes_ThreeWords - In up to three words, how does this image make you feel? Open text
# Gstripes_Means - What does the image mean? Open text
# GS_EmoAnxious - How far do you feel that this image makes you feel: (Anxious scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoHopeful - How far do you feel that this image makes you feel: (Hopeful scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoInspired - How far do you feel that this image makes you feel: (Inspired scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoHappy - How far do you feel that this image makes you feel: (Happy scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoAngry - How far do you feel that this image makes you feel: (Angry scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoSad - How far do you feel that this image makes you feel: (Sad scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoGuilty - How far do you feel that this image makes you feel: (Guilty scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoHopeless - How far do you feel that this image makes you feel: (Hopeless scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree 
# GS_EmoInformed - How far do you feel that this image makes you feel: (Informed scale) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree

# GS_ImageUnderstand - How far do you agree with (I understand what the image means) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_ImageUnderstandCC - How far do you agree with (The image helps me to understand climate change/global warming) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_ImageAct - How far do you agree with (The image inspires me to act on climate change/ global warming) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_Image_NoUnderstand - How far do you agree with (I do not understand what the image means) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_Image_Attention - How far do you agree with (This image caught my attention) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_Image_VisualAppeal - How far do you agree with (The image looks visually appealing) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_Image_ShareIntend - How far do you agree with (I would share the image myself in the future ) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree
# GS_Image_LikeColour - How far do you agree with (I like the colours of the image) Likert 5 points, 1=Strongly Disagree, 5=Strongly Agree

# Demographics
# Gender - Male(1)/Female(2)/Non-Binary(3)/Other(4)
# Age - Open text
# Country - Country the survey was carried out in US/UK
# Country_Resident - Assumed to be "do you live in US/UK" with 1/2 response. Assumed to be a match to Country column.
# Ethnicity - Asian(1)/Black(2)/Mixed(3)/White(4)/Other(5)/Prefer not to say(6)/Hispanic(7, US only)  [Note this is different for US/UK but has the same pattern]
# What is your ethnicity? [other, please describe] - Open text
# Education - What is your highest level of education? Unclear scale, and only for the US data.
# Education [other, please describe] - Unclear formatting for US responses. Seems to be another likert rather than an open text.
# Political_Orientation - How would you best describe your political orientation? Strongly Liberal(1)/Somewhat liberal(2)/Neutral(3)/Somewhat conservative(4)/Strongly Conservative(5)
# Political_Party_UK - What political party are you most likely to vote for in an upcoming election? Labour(1)/Conservatives(2)/Lib Dem(3)/Green(4)/Reform(5)/Ind(6)/Other(7)
# Political_Party-UK-Other - Please describe Open text
# Political_Party_US - What political party are you most likely to vote for in an upcoming election? Democrats(1)/Republicans(2)/Libertarian(3)/Green(4)/Constitution(5)/Natural Law(6)/Other(7)
# Political_Party-US-Other - Please describe Open text
# Mental-Physical_Conditions - Do you consider yourself to have any chronic mental or physical health issues or conditions? Yes(1)/No(2)/Prefer not answer(3)

# CC_InformationSeek - How often, in a typical week, do you intentionally seek out information about climate change? Not at all (1)/Sometimes[1-3 days per week](2)/Often[4-6 days per week](3)/Everyday[7 days per week](4)
# CC_Anthropogenic - Do you think that climate change is caused by natural processes, human activity, or both? Entirely natural(1)/Mainly natural(2)/Equally natural and human(3)/Mainly human(4)/Entirely human(5)/Not happening(6)
# CC_ImpactExperience - In the last 5 years have you experienced any form of flood, storm, heatwave or drought damage (including to your home and garden)? Once(1)/2-3 times(2)/4+ times(3)/No(4)
# CC_Worry - Question unknown, likely about personal level of worry about climate change - Not at all(1)/Not very(2)/Somewhat(3)/Very(4)/Extremely(5)

# Final_Comments - Unclear question, likely asking for any other thoughts. Open text

# Clean the columns.
df = df[df['Response Status']=='Completed']  ## Keep only the finished surveys.
df['Political_Orientation'] = df['Political_Orientation'].astype(float)

# Age is free text, so coerce to numeric rather than assuming it already is - non-numeric entries become NaN.
age_numeric = pd.to_numeric(df['Age'], errors='coerce')
non_numeric_age = (age_numeric.isna() & df['Age'].notna()).sum()
if non_numeric_age:
	print(f'{non_numeric_age} Age value(s) could not be parsed as numbers and were treated as missing')
df['Age'] = age_numeric

############ BASIC COUNTING ########################

# Check for any rows where consent is not 1.
for i in range(1,10):
	assert len(df[df[f'Consent_{i}'] != 1]) == 0, f'One of the rows has Consent_{i} != 1'
print('Passed consent checks')

# Check for unique Prolific IDs.
assert len(df[df['Prolific_ID'].duplicated()]) == 0, 'There are duplicated Prolific IDs'
print('Passed Prolific ID checks')

# Check for following 3 word instruction on stripes and Gstripes.
bad_rows = [r for r in df['Stripes_Three-Words'] if type(r) == str and len(r.split()) > 3]
print(f'Number of responses with >3 words about climate stripes: {len(bad_rows)}')
bad_rows = [r for r in df['Gstripes_ThreeWords'] if type(r) == str and len(r.split()) > 3]
print(f'Number of responses with >3 words about other stripes: {len(bad_rows)}')

# Check for anyone saying "I didn't see the stripes" in details, but still ticking yes to one of the categories (at least one of these is speculating on where it could have come from)
# Note this is a crude measure and has counted people saying "I didn't see people change their..."
didnt_see_mask = [type(r) == str and ('not see' in r.lower() or "didn't see" in r.lower()) for r in df['Sseen_Details']]
print(f"Number of times people said they hadn't seen the stripes before: {sum(didnt_see_mask)}")

# Check for complimentary answers to Image_UnderstandMean and Image_NoUnderstand
# The intuition here is they should add to around 6, want to flag any that are lower or higher
print(f'Low sums of understanding and not understanding the image: {sum((df['Image_UnderstandMean']+df['Image_NoUnderstand']) <5)}')
print(f'High sums of understanding and not understanding the image: {sum((df['Image_UnderstandMean']+df['Image_NoUnderstand']) >7)}')

print(f'Low sums of understanding and not understanding the biodiversity image: {sum((df['GS_ImageUnderstand']+df['GS_Image_NoUnderstand']) <5)}')
print(f'High sums of understanding and not understanding the biodiversity image: {sum((df['GS_ImageUnderstand']+df['GS_Image_NoUnderstand']) >7)}')

# Check for any invalid ages (excluding empty)
print(f'Number of NaNs in Age: {sum(df['Age'].isna())}')

# Count by gender, political leaning, age.
for i,lab in enumerate(['Male','Female','Non-Binary','Other']):
	print(f'Number of {lab} respondants: {df['Gender'].value_counts().get(i+1, 0)}')

for i in range(1,10):
	print(f"Number of respondants {10*i}<=age<={10*i+9}: {len(df[df['Age'].between(i*10,i*10+9)])}")

for i,lab in enumerate(['Strongly Liberal','Somewhat liberal','Neutral','Somewhat conservative','Strongly Conservative']):
	print(f'Number of {lab} respondants: {df['Political_Orientation'].value_counts().get(i+1, 0)}')

# Count simple responses - who had seen before and where, whether it is understood.
for i,lab in enumerate(['Yes','No','Unsure']):
	print(f'Number of answers seen before {lab}: {df['Stripes_Seen'].value_counts().get(i+1, 0)}')

for lab in ['SocialMedia','Museum','Community','Business','Sport','Music','News','Weather','Building','Clothing','Transport','Book','Politics','Education']:
	print(f'Number of answers seen before {lab}: {df['Sseen_'+lab].sum()}')

for i,lab in enumerate(['Yes','No','Unsure']):
	print(f'Number of answers GStripes seen before {lab}: {df['GStripes_Seen'].value_counts().get(i+1, 0)}')

########### STATS MODELS  ##########################

# Run an OrderedModel of response on a given likert question to population demographics.
# Supported with Claude Sonnet 5 medium effort on 29/7/26.
categorical_predictors = ['Gender', 'Country', 'Ethnicity'] # dummy-coded predictors
continuous_predictors = ['Age', 'Political_Orientation']               # standardized, kept continuous

# Predictor cleaning/encoding is the same for every question below (it doesn't depend on response_col),
# so build it once here rather than redoing the copy/dropna/dummy-coding/z-scoring on every loop pass.
df_reg = df.copy()
df_reg = df_reg.dropna(subset=categorical_predictors)  # drop missing Gender/Country/Ethnicity now, before dummy-coding hides them
df_reg['Gender'] = df_reg['Gender'].astype(str)
df_reg['Ethnicity'] = df_reg['Ethnicity'].astype(str)

X_parts = []

# categorical predictors -> dummy coded, drop_first avoids collinearity
if categorical_predictors:
	X_cat = pd.get_dummies(df_reg[categorical_predictors], drop_first=True).astype(float)
	X_parts.append(X_cat)

# continuous predictors -> standardized (z-score) so coefficients are comparable
if continuous_predictors:
	X_cont = df_reg[continuous_predictors].astype(float).copy()
	for col in continuous_predictors:
		X_cont[col] = (X_cont[col] - X_cont[col].mean()) / X_cont[col].std()
	X_parts.append(X_cont)

X_base = pd.concat(X_parts, axis=1)

results = []
fitted_models = {}  # column to model
X_per_question = {}  # column to X
for response_col in ['Emo_Anxious','Emo_Hopeful','Emo_Inspired','Emo_Happy','Emo_Angry','Emo_Sad','Emo_Guilty','Emo_Hopeless','Emo_Informed','GS_EmoAnxious','GS_EmoHopeful','GS_EmoInspired','GS_EmoHappy','GS_EmoAngry','GS_EmoSad','GS_EmoGuilty','GS_EmoHopeless','GS_EmoInformed','Image_UnderstandMean','Image_UnderstandCC','Image_Attention','Image_WouldShare','Image_ActCC']:             # your 1-5 Likert column
	categories_in_order = sorted(df_reg[response_col].dropna().unique())  # e.g. [1,2,3,4,5]

	response = pd.Series(
		pd.Categorical(df_reg[response_col], categories=categories_in_order, ordered=True),
		index=df_reg.index, name=response_col
	)

	assert response.cat.ordered, "Response column did not convert to ordered categorical"

	# ------------------------------------------------------------------
	# 4. DROP rows with any missing values (OrderedModel can't handle NaNs)
	# ------------------------------------------------------------------
	model_df = pd.concat([response, X_base], axis=1).dropna()
	y = model_df[response_col]
	X = model_df.drop(columns=[response_col])

	#print(f"Fitting on {len(model_df)} rows, {X.shape[1]} predictors")

	# ------------------------------------------------------------------
	# 5. FIT the ordinal logistic regression (proportional odds model)
	# ------------------------------------------------------------------
	mod = OrderedModel(y, X, distr='logit')
	res = mod.fit(method='bfgs', disp=False)

	#print(res.summary())

	# ------------------------------------------------------------------
	# 6. QUICK SANITY CHECKS
	# ------------------------------------------------------------------
	#print(f"\nPseudo R-squared: {res.prsquared:.4f}")
	#print(f"Converged: {res.mle_retvals['converged']}")

	params = res.params
	conf_int = res.conf_int()

	summary = pd.DataFrame({
		'predictor': params.index,
		'coef': params.values,
		'std_err': res.bse.values,
		'p_value': res.pvalues.values,
		'ci_low': conf_int[0].values,
		'ci_high': conf_int[1].values,
	})

	# drop threshold rows (labeled like '1/2', '2/3', etc.)
	summary = summary[~summary['predictor'].astype(str).str.contains('/')]

	summary['question'] = response_col
	summary['n'] = len(X)
	summary['odds_ratio'] = np.exp(summary['coef'])

	results.append(summary)
	fitted_models[response_col] = res
	X_per_question[response_col] = X

# Collate, correct and present the results
all_results = pd.concat(results, ignore_index=True)

import pandas as pd
import numpy as np

def predicted_prob_shift(res, X, predictor, target_category=None, comparison_value=1, baseline_value=0):
    """
    Compute the percentage-point shift in predicted probability for a one-unit
    change in `predictor`, holding all other predictors at their mean.

    Works for OrderedModel (ordinal), MNLogit (multinomial), and Logit (binary).

    Parameters
    ----------
    res : fitted statsmodels results object
    X : the predictor matrix used to fit the model (for column names + means)
    predictor : str, name of the column to vary
    target_category : the outcome category to report the shift for.
        - For OrderedModel/MNLogit: the category label (e.g. 3, or 'Anxious')
        - For Logit: leave as None (there's only one probability)
    comparison_value : value to set `predictor` to in the "scenario" row (default 1)
    baseline_value : value to set `predictor` to in the "baseline" row (default 0)

    Returns
    -------
    dict with baseline_prob, scenario_prob, and pp_shift (percentage points)
    """
    # start both rows at the mean of every predictor
    baseline = X.mean().to_frame().T
    scenario = X.mean().to_frame().T

    baseline[predictor] = baseline_value
    scenario[predictor] = comparison_value

    pred_baseline = res.predict(baseline)
    pred_scenario = res.predict(scenario)

    # Logit returns a single probability per row (Series/array)
    if target_category is None:
        p_base = float(np.asarray(pred_baseline).flatten()[0])
        p_scen = float(np.asarray(pred_scenario).flatten()[0])
    # OrderedModel/MNLogit return a probability per category
    else:
        p_base = float(pred_baseline[target_category].iloc[0])
        p_scen = float(pred_scenario[target_category].iloc[0])

    return {
        'predictor': predictor,
        'target_category': target_category,
        'baseline_prob': round(p_base, 4),
        'scenario_prob': round(p_scen, 4),
        'pp_shift': round((p_scen - p_base) * 100, 2)
    }

# FDR correction across ALL tests in the batch, not per-question. We skip any rows that are nans as they cause the correction to fail.
valid_mask = all_results['p_value'].notna()

all_results['p_adj_fdr'] = np.nan  # start all as nan
all_results.loc[valid_mask, 'p_adj_fdr'] = multipletests(
    all_results.loc[valid_mask, 'p_value'], method='fdr_bh'
)[1]

all_results = all_results[['question', 'predictor', 'n', 'coef', 'odds_ratio', 
							'std_err', 'p_value', 'p_adj_fdr', 'ci_low', 'ci_high']]

# tidy column names and order
all_results = all_results.rename(columns={
	'Coef.': 'coef', 'Std.Err.': 'std_err', 'P>|z|': 'p_value',
	'[0.025': 'ci_low', '0.975]': 'ci_high'
	})[['question', 'predictor', 'n', 'coef', 'odds_ratio', 'std_err', 
	'p_value', 'p_adj_fdr', 'ci_low', 'ci_high']]

all_results['significant_fdr'] = all_results['p_adj_fdr'] < 0.05

# which predictors survive across the most questions?
all_results[all_results['significant_fdr']].groupby('predictor').size().sort_values(ascending=False)

significant_rows = all_results[all_results['significant_fdr']]

grid = (
    all_results
    .assign(significant_fdr=all_results['p_adj_fdr'] < 0.05)
    .pivot(index='predictor', columns='question', values='significant_fdr')
)

# order predictors by how many questions they're significant for
grid['n_significant'] = grid.sum(axis=1)
grid = grid.sort_values('n_significant', ascending=False)

print(grid)

# Look at where different predictors are significant and rank by odds ratio.
summary_by_predictor = (
    all_results
    .groupby('predictor')
    .agg(
        n_significant=('significant_fdr', 'sum'),
        median_odds_ratio=('odds_ratio', 'median'),
        median_p_adj=('p_adj_fdr', 'median')
    )
    .sort_values('n_significant', ascending=False)
)
print(summary_by_predictor)

####################  SURVEY SEEN BEFORE ANALYSIS   ###############
response_col = 'Stripes_Seen'
categorical_predictors = ['Gender', 'Country', 'Ethnicity'] # dummy-coded predictors
continuous_predictors = ['Age', 'Political_Orientation']               # standardized, kept continuous

df_reg = df.copy()
df_reg = df_reg.dropna(subset=categorical_predictors)  # drop missing Gender/Country/Ethnicity now, before dummy-coding hides them
df_reg['Stripes_Seen'] = df_reg['Stripes_Seen'].replace(3,2)  # Tidying this to make the model easier, treat "don't know" as "no"
df_reg['Stripes_Seen'] = df_reg['Stripes_Seen'].map({1: 1.0, 2: 0.0})  # sm.Logit needs a numeric 0/1 target, not a categorical: 1 = Yes seen before, 0 = No/Unsure. map() (not ==) so missing responses stay NaN and get dropped below, rather than being coded as "No".
df_reg['Gender'] = df_reg['Gender'].astype(str)
df_reg['Ethnicity'] = df_reg['Ethnicity'].astype(str)

# ------------------------------------------------------------------
# 3. BUILD predictor matrix X
# ------------------------------------------------------------------
X_parts = []

# categorical predictors -> dummy coded, drop_first avoids collinearity
if categorical_predictors:
	X_cat = pd.get_dummies(df_reg[categorical_predictors], drop_first=True).astype(float)
	X_parts.append(X_cat)

# continuous predictors -> standardized (z-score) so coefficients are comparable
if continuous_predictors:
	X_cont = df_reg[continuous_predictors].astype(float).copy()
	for col in continuous_predictors:
		X_cont[col] = (X_cont[col] - X_cont[col].mean()) / X_cont[col].std()
	X_parts.append(X_cont)

X = pd.concat(X_parts, axis=1)

# ------------------------------------------------------------------
# 4. DROP rows with any missing values (OrderedModel can't handle NaNs)
# ------------------------------------------------------------------
model_df = pd.concat([df_reg[response_col], X], axis=1).dropna()
y = model_df[response_col]
X = model_df.drop(columns=[response_col])

#print(f"Fitting on {len(model_df)} rows, {X.shape[1]} predictors")

# ------------------------------------------------------------------
# 5. FIT the ordinal logistic regression (proportional odds model)
# ------------------------------------------------------------------
X = sm.add_constant(X)
mod = sm.Logit(y, X)
res = mod.fit(method='bfgs', disp=False)

#print(res.summary())

# ------------------------------------------------------------------
# 6. QUICK SANITY CHECKS
# ------------------------------------------------------------------
#print(f"\nPseudo R-squared: {res.prsquared:.4f}")
#print(f"Converged: {res.mle_retvals['converged']}")

params = res.params
conf_int = res.conf_int()

summary = pd.DataFrame({
	'predictor': params.index,
	'coef': params.values,
	'std_err': res.bse.values,
	'p_value': res.pvalues.values,
	'ci_low': conf_int[0].values,
	'ci_high': conf_int[1].values,
})

# drop threshold rows (labeled like '1/2', '2/3', etc.)
summary = summary[~summary['predictor'].astype(str).str.contains('/')]

summary['question'] = response_col
summary['n'] = len(X)
summary['odds_ratio'] = np.exp(summary['coef'])

print('Model results for demographic influence on having seen before')
print(summary)

################   SURVEY THEMATIC ANAYLSIS  ###################
# Here we calculate whether demographics influence the likelihood of people describing the stripes accurately.
# Load the labelled data.
df_labelled = pd.read_excel('',engine='openpyxl')

# Fill nans with 0s, replace 2s with 1s.
df_labelled['human_label'] = df_labelled['human_label'].fillna(0)
df_labelled['human_label'] = df_labelled['human_label'].replace(2,1)

# Add the ethnicity data back in. Use a map (not a dict lookup) since df_labelled may have been
# hand-labelled from an older export and can contain Response IDs no longer present in df -
# those rows get NaN Ethnicity here and are dropped later by the dropna(subset=categorical_predictors) below.
eths = {r:e for r,e in zip(df['Response ID'],df['Ethnicity'])}
df_labelled['Ethnicity'] = df_labelled['Response ID'].map(eths)

unmatched = df_labelled['Ethnicity'].isna().sum()
if unmatched:
	print(f'{unmatched} labelled response(s) have no matching Response ID in df - Ethnicity left blank and they will be dropped from this model')

response_col = 'human_label'
categorical_predictors = ['Gender', 'Country', 'Ethnicity'] # dummy-coded predictors
continuous_predictors = ['Age', 'Political_Orientation']               # standardized, kept continuous

df_reg = df_labelled

# Other cleaning
df_reg = df_reg.dropna(subset=categorical_predictors)  # drop missing Gender/Country/Ethnicity now, before dummy-coding hides them
df_reg['Gender'] = df_reg['Gender'].astype(str)
df_reg['Ethnicity'] = df_reg['Ethnicity'].astype(str)

# ------------------------------------------------------------------
# 3. BUILD predictor matrix X
# ------------------------------------------------------------------
X_parts = []

# categorical predictors -> dummy coded, drop_first avoids collinearity
if categorical_predictors:
	X_cat = pd.get_dummies(df_reg[categorical_predictors], drop_first=True).astype(float)
	X_parts.append(X_cat)

# continuous predictors -> standardized (z-score) so coefficients are comparable
if continuous_predictors:
	X_cont = df_reg[continuous_predictors].astype(float).copy()
	for col in continuous_predictors:
		X_cont[col] = (X_cont[col] - X_cont[col].mean()) / X_cont[col].std()
	X_parts.append(X_cont)

X = pd.concat(X_parts, axis=1)

# ------------------------------------------------------------------
# 4. DROP rows with any missing values (OrderedModel can't handle NaNs)
# ------------------------------------------------------------------
model_df = pd.concat([df_reg[response_col], X], axis=1).dropna()
y = model_df[response_col]
X = model_df.drop(columns=[response_col])

#print(f"Fitting on {len(model_df)} rows, {X.shape[1]} predictors")

# ------------------------------------------------------------------
# 5. FIT a logit model (can't use ordinal since the outcome here is binary)
# ------------------------------------------------------------------
X = sm.add_constant(X)
mod = sm.Logit(y, X)
res = mod.fit(method='bfgs', disp=False)

#print(res.summary())

# ------------------------------------------------------------------
# 6. QUICK SANITY CHECKS
# ------------------------------------------------------------------
#print(f"\nPseudo R-squared: {res.prsquared:.4f}")
#print(f"Converged: {res.mle_retvals['converged']}")

params = res.params
conf_int = res.conf_int()

summary = pd.DataFrame({
	'predictor': params.index,
	'coef': params.values,
	'std_err': res.bse.values,
	'p_value': res.pvalues.values,
	'ci_low': conf_int[0].values,
	'ci_high': conf_int[1].values,
})

# drop threshold rows (labeled like '1/2', '2/3', etc.)
summary = summary[~summary['predictor'].astype(str).str.contains('/')]

summary['question'] = response_col
summary['n'] = len(X)
summary['odds_ratio'] = np.exp(summary['coef'])

print('Model results for demographic influence on getting the right description')
print(summary)

############### TEXT ANALYSIS ######################
# Make wordclouds of what 3 words by demographics (country, political leaning, age etc)
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def plot_column_wordcloud(df,col_to_plot,col_to_filter,label):
	""" Produce a wordlcoud of a given column in the dataframe, concatenating all text in the column. """

	text = ' '.join([t for t in df[col_to_plot] if type(t)==str]) # Need to skip any empty rows that are nans etc.

	wc = WordCloud(
		width=800,
		height=400,
		background_color="white",
		max_words=200,
		colormap="viridis"
	).generate(text)

	plt.figure(figsize=(10, 5))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.savefig(f'figs/{col_to_filter}_{label}_{col_to_plot}_cloud.png', dpi=300, bbox_inches="tight")
	plt.close()

# Plot everything together.
plot_column_wordcloud(df,'Stripes_Three-Words','all','all')

# Country
for country in ['UK','US']:
	df_trim  = df[df['Country']==country]

	plot_column_wordcloud(df_trim,'Stripes_Three-Words','Country',country)

# Political orientation
for orientation in range(1,6):
	df_trim = df[df['Political_Orientation']==orientation]

	plot_column_wordcloud(df_trim,'Stripes_Three-Words','Political_Orientation',orientation)

# Gender
for gender in range(1,5):
	df_trim = df[df['Gender']==gender]

	plot_column_wordcloud(df_trim,'Stripes_Three-Words','Gender',gender)

# Do the other stripes seen.
df_trim = df[~df['Sseen_Other'].isna()]
plot_column_wordcloud(df_trim,'Sseen_Other','','')

# Run tf-idf comparisons between the demographics (i.e. salient words for each group).
def plot_tfidf_wordclouds(df,text_col,group_col):

	# --- 3. Build one "document" per group ---
	grouped = df.groupby(group_col)[text_col].apply(lambda texts: " ".join(texts.dropna().astype(str)))
	group_names = grouped.index.tolist()
	docs = grouped.values.tolist()

	# --- 4. Stopwords ---
	custom_stopwords = STOPWORDS.union({"nan"})  # add domain-specific junk here

	# --- 5. TF-IDF across groups ---
	vectorizer = TfidfVectorizer(
		stop_words=list(custom_stopwords),
		lowercase=True,
		ngram_range=(1, 2),
		min_df=1,
		max_features=5000
	)

	tfidf_matrix = vectorizer.fit_transform(docs)  # shape: (n_groups, n_terms)
	feature_names = vectorizer.get_feature_names_out()

	# --- 6. Plot a wordcloud per group ---
	n_groups = len(group_names)
	n_cols = min(3, n_groups)
	n_rows = int(np.ceil(n_groups / n_cols))

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
	axes = np.array(axes).reshape(-1)  # flatten regardless of shape

	for i, name in enumerate(group_names):
		row = tfidf_matrix[i].toarray().flatten()
		freqs = {feature_names[j]: row[j] for j in np.nonzero(row)[0]}

		wc = WordCloud(
			width=600,
			height=400,
			background_color="white",
			colormap="viridis",
			max_words=100
		).generate_from_frequencies(freqs)

		axes[i].imshow(wc, interpolation="bilinear")
		axes[i].axis("off")
		axes[i].set_title(name, fontsize=14)

	# hide any unused subplot axes
	for j in range(len(group_names), len(axes)):
		axes[j].axis("off")

	plt.tight_layout()
	plt.savefig(f'figs/tfidf_{group_col}_{text_col}_cloud.png', dpi=300, bbox_inches="tight")
	plt.close()

# Split by demographics
plot_tfidf_wordclouds(df,"Stripes_Three-Words","Country")
plot_tfidf_wordclouds(df,"Stripes_Three-Words","Gender")
plot_tfidf_wordclouds(df,"Stripes_Three-Words","Political_Orientation")
plot_tfidf_wordclouds(df,"Stripes_Three-Words","Ethnicity")

# Run a 2D PCA plot of the descriptions of what the texts show (ideally do the before and after reveals on the same PCA axes)
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

model = "all-MiniLM-L6-v2"
min_size = 25
embedder = SentenceTransformer(model)  # load once - reused across every fit_topics() call below
def fit_topics(df,text_col,model,min_size,embedder):
	""" Fit the topic model over a given column, using the model and parameters specified. """

	docs = df[text_col].dropna().astype(str).tolist()
	valid_idx = df[text_col].dropna().index
	df_valid = df.loc[valid_idx].reset_index(drop=True)

	# --- 2. Embed with all-MiniLM-L6-v2 ---
	embeddings = None
	try:
		with open(f'{text_col}_{model}_embeddings.np','rb') as f:
			embeddings = np.load(f)
		if len(embeddings) != len(docs):
			print(f'Cached embeddings for {text_col} cover {len(embeddings)} docs but current data has {len(docs)} - recomputing.')
			embeddings = None
	except (EOFError, ValueError, OSError) as e:
		print(f'Could not load cached embeddings for {text_col} ({e}) - recomputing.')

	if embeddings is None:
		embeddings = embedder.encode(docs, show_progress_bar=True)  # shape: (n_docs, 384)

		with open(f'{text_col}_{model}_embeddings.np','wb') as f:
			np.save(f,embeddings)

	vectorizer_model = CountVectorizer(stop_words="english")

	# --- 3. Fit BERTopic using precomputed embeddings ---
	# fix random_state here so topic assignments are reproducible across runs
	topic_umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
	topic_model = BERTopic(embedding_model=embedder, umap_model=topic_umap_model, vectorizer_model=vectorizer_model, min_topic_size=min_size)
	topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

	df_valid["noisy topic"] = topics

	# Run noise reclassification.
	new_topics = topic_model.reduce_outliers(
    docs, topics,
    strategy="embeddings",
    threshold=0.4,
    embeddings=embeddings,  # the same full array you passed to fit_transform
	)
	topic_model.update_topics(
    docs,
    topics=new_topics,
    vectorizer_model=topic_model.vectorizer_model,
    ctfidf_model=topic_model.ctfidf_model,
    representation_model=topic_model.representation_model,
	)
	
	topics = topic_model.topics_

	df_valid["topic"] = topics
	topic_info = topic_model.get_topic_info()
	topic_label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
	df_valid["topic_label"] = df_valid["topic"].map(topic_label_map)

	# Save the topic info.
	topic_info.to_csv(f'{text_col}_{model}_min_size_{min_size}_topic_info_reclassification.csv')
	df_valid.drop(columns=[c for c in df_valid.columns if c not in ['Response ID', 'Country', 'Stripes_Seen', 'Stripes_Three-Words', text_col, 'Gender', 'Age', 'Ethnicity', 'Political_Orientation', 'topic_label']]).to_csv(f'{text_col}_{model}_min_size_{min_size}_topic_labelled_data_reclassification.csv')

	# --- 4. UMAP projection for visualization (separate 2D reducer) ---
	# note: this is a distinct UMAP instance from the one BERTopic used internally —
	# that one reduced to 5D for clustering; this one reduces to 2D purely for plotting
	viz_umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
	proj = viz_umap_model.fit_transform(embeddings)
	df_valid["umap_x"] = proj[:, 0]
	df_valid["umap_y"] = proj[:, 1]

	# Calculate the similarities of the texts with the description of the Stripes shown.
	stripes_desc = "This graphic is a visual representation of the change in global temperature since 1850. Each stripe or bar represents the global average temperature for that year."
	desc_emb = embedder.encode(stripes_desc)

	desc_sims = cosine_similarity(desc_emb.reshape(1, -1), embeddings)

	return df_valid, desc_sims

df_valid, pre_sims = fit_topics(df,'Stripes_Shows',model,min_size,embedder)
df_valid_post_prompt, post_sims = fit_topics(df,'Sshows_PostPrompt',model,min_size,embedder)

from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(post_sims.flatten(), pre_sims.flatten(), alternative='greater')

print(f'Mann Whitney U for post sims are greater than pre sims: {stat}, {p_value}')

# --- 5. Plot, colored by topic or by an arbitrary column ---
def plot_projection(df_plot, label,color_col="topic_label", title=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    categories = df_plot[color_col].astype(str).unique()
    cmap = plt.get_cmap("tab20", len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}

    for cat in categories:
        subset = df_plot[df_plot[color_col].astype(str) == cat]
        ax.scatter(
            subset["umap_x"], subset["umap_y"],
            label=cat, color=color_map[cat],
            s=30, alpha=0.7, edgecolors="none"
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP projection colored by {color_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(f"figs/{label}_umap_{color_col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# default: color by BERTopic's topic label, filtering out junk
plot_projection(df_valid[df_valid["topic"] != -1], 'Stripes_Shows', color_col="topic_label")

plot_projection(df_valid, 'Stripes_Shows', color_col="Gender")
plot_projection(df_valid, 'Stripes_Shows', color_col="Political_Orientation")

code.interact(local=locals())