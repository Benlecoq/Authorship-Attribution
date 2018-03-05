
# coding: utf-8

# In[125]:


import pandas as pd
import os
from os import walk

#Data loading, formatting, label creation

# Create local path
path = os.getcwd()
print(path)

# Create dataframe from all text files in directory. 
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)    
df = pd.DataFrame(columns=['Artist','SongName', 'Lyrics']);


# Split the file name to populate column headers 
inc = 0;
for filename in f:
    valuesArr = filename.split("_");
    if(len(valuesArr) != 2):
        print("Found invalid name "+filename);
        continue;
    file = open(path+filename,'r', encoding = "ISO-8859-1");
    message = file.read();
    df.loc[inc] = [valuesArr[0],valuesArr[1],message]
    inc = inc+1;
    
df['SongName'] = df['SongName'].map(lambda x: x.rstrip('.txt')) # Strip off .txt

print (df.head(3))


# In[126]:


# Create functions to add time period and Location labels from Artist

def to_period(x):
    if x == 'biggy' or 'pac':
        return '90s'
    elif x =='joey' or "kendrick":
        return 'now'
    else:
        return 'Undefined'

def to_location(x):
    if x == 'biggy' or "joey":
        return 'east'
    elif x == 'pac' or "kendrick":
        return 'west'
    else:
        return 'Undefined'

# Add output to dataframe
df['TimePeriod'] = df.Artist.apply(to_period)
df['Location'] = df.Artist.apply(to_location)

#Rearrange dataframe in cleaner order
cols = ['Artist','TimePeriod','Location','SongName','Lyrics']
df = df[cols]

print(df.head(3))


# In[127]:


import re
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Text Preprocessing

# Remove punctuation
def punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') 
    return text

# Lower case
def lowercase (text): 
    txt = text.lower() 
    return txt

# Lemmatize
def lemmatize(text): 
    tokens = word_tokenize(text)
    lemma=WordNetLemmatizer()
    test=[lemma.lemmatize(word) for word in tokens]
    final = ' '.join(test)
    return final

df['CleanLyrics'] = df.Lyrics.apply(punctuation)
df['CleanLyrics'] = df.CleanLyrics.apply(lowercase)
df['CleanLyrics'] = df.CleanLyrics.apply(lemmatize)


print(df.head(3))


# In[128]:


from nltk import pos_tag
from nltk.util import ngrams

# Extract Syntactic features (Part-of-Speech tags)

# Function to POS tag text
def postag(string):
    tokens = word_tokenize(string)
    tokens_pos = pos_tag(tokens)
    join = ' '.join([pos for word, pos in tokens_pos])
    return (join)

# Function to POS tag and attach to original word (POSTAG_WORD)
def posandword(string):
    tokens = word_tokenize(string)
    tokens_pos = pos_tag(tokens)
    join = ' '.join([word + '_' + pos for word, pos in tokens_pos])
    return (join)

df['PosTag'] = df.CleanLyrics.apply(postag)
df['PosWord'] = df.CleanLyrics.apply(posandword)

print(df.head(3))


# In[129]:


# Tag text according to Regressive Imagery Dictionary (RID) Categories

# Categories and corresponding words
emotions = ['RidEmotions', 'emotionspositiveaffect', 'amus', 'amusement', 'blith', 'carefre', 'celebrat', 'cheer', 'cheerful', 'cheery', 'chuckl', 'delight', 'delightful', 'elat', 'enjoy', 'enjoyabl', 'enjoyment', 'entertain', 'entertainment', 'enthusiasm', 'enthusiastic', 'excit', 'exhilerat', 'exult', 'exultant', 'fun', 'funny', 'gaiety', 'gay', 'glad', 'gladnes', 'glee', 'gleeful', 'gleely', 'gratifi', 'gratify', 'grin', 'happines', 'happy', 'hilariou', 'humor', 'humorou', 'humour', 'humourou', 'jocund', 'jok', 'jolly', 'jovial', 'joy', 'joyful', 'joyou', 'laugh', 'laughter', 'merriment', 'merry', 'mirth', 'mirthful', 'overjoy', 'playful', 'pleasantry', 'pleasur', 'pleasurabl', 'rejoic', 'relief', 'reliev', 'rollick', 'satisf', 'smil', 'thril', 'thrill', 'vivaciou', 'vivacity', 'emotionsanxiety', 'tremor', 'afraid', 'aghast', 'alarm', 'anguish', 'anxi', 'avoid', 'blush', 'care', 'coward', 'cower', 'crisi', 'dangerou', 'desperat', 'distres', 'dread', 'dreadful', 'fear', 'fearful', 'frantic', 'fret', 'fright', 'horrifi', 'horrify', 'horror', 'nervou', 'nervousnes', 'panic', 'phobia', 'phobic', 'scare', 'scared', 'scare', 'scary', 'shriek', 'shudder', 'shy', 'terrifi', 'terrify', 'terror', 'timid', 'trauma', 'trembl', 'tremulou', 'troubl', 'uneasines', 'uneasy', 'worri', 'worry', 'emotionssadness', 'aggrieved', 'ala', 'deject', 'depres', 'depress', 'despair', 'despondant', 'despondent', 'dirg', 'disappoint', 'disappointment', 'disconsolat', 'discourag', 'dishearten', 'dismal', 'dissatisfi', 'dissatisfy', 'distraught', 'doldrum', 'downcast', 'dreary', 'elegy', 'forlorn', 'frown', 'funereal', 'grie', 'groan', 'hopeles', 'humiliat', 'lament', 'lamentat', 'lone', 'lonelines', 'melanc', 'miserabl', 'miseri', 'misery', 'moan', 'mourn', 'mournful', 'orphan', 'pain', 'pitiful', 'plaint', 'regret', 'regretful', 'remors', 'repent', 'repentanc', 'repentenc', 'rue', 'sad', 'sadden', 'sadly', 'sadnes', 'sob', 'sobb', 'sob', 'somber', 'sombr', 'sorrow', 'sorrowful', 'sorry', 'suffer', 'tearful', 'tragedy', 'tragic', 'unhappines', 'unhappy', 'wail', 'weep', 'wept', 'whin', 'woe', 'woe', 'emotionsaffection', 'affect', 'affectionat', 'amorou', 'amourou', 'appreciat', 'attractiv', 'befriend', 'belov', 'bosom', 'bridal', 'bride', 'cherish', 'congenial', 'cordial', 'courtship', 'darl', 'dear', 'devot', 'embrac', 'enamor', 'enamour', 'endear', 'familiar', 'fonder', 'farewell', 'favor', 'favour', 'fianc', 'flirt', 'fond', 'fondnes', 'fraternity', 'friend', 'friendship', 'goodby', 'grateful', 'intimacy', 'intimat', 'kind', 'kindnes', 'like', 'liking', 'lov', 'marri', 'marriag', 'marry', 'mate', 'mated', 'mate', 'mating', 'mercy', 'pat', 'pat', 'patt', 'piti', 'pity', 'romanc', 'sweetheart', 'sympat', 'unselfish', 'warmheart', 'welcom', 'wooed', 'wooing', 'woos', 'emotionsaggression', 'abhor', 'abus', 'abusiv', 'accus', 'afflict', 'aggress', 'aggressiv', 'ambush', 'anger', 'angri', 'angrier', 'angry', 'annihilat', 'annoy', 'annoyanc', 'antagoniz', 'argu', 'argument', 'army', 'arrow', 'assault', 'attack', 'aveng', 'ax', 'axe', 'ax', 'battl', 'beak', 'beat', 'beaten', 'betray', 'blade', 'blam', 'bloody', 'bother', 'brawl', 'break', 'brok', 'broken', 'brutal', 'cannon', 'chid', 'combat', 'complain', 'conflict', 'condemn', 'controversy', 'critic', 'cruel', 'crush', 'cut', 'cut', 'cutt', 'damag', 'decei', 'defeat', 'degrad', 'demolish', 'depriv', 'derid', 'despis', 'destroy', 'destruct', 'destructiv', 'detest', 'disagre', 'disagreement', 'disapprov', 'discontent', 'dislik', 'disput', 'disturb', 'doubt', 'enemi', 'enemy', 'enrag', 'exasperat', 'controversial', 'critique', 'disparag', 'irritable', 'exploit', 'exterminat', 'feud', 'fierc', 'fight', 'fought', 'furiou', 'fury', 'gash', 'grappl', 'growl', 'grudg', 'gun', 'gunn', 'gun', 'harm', 'harsh', 'hate', 'hatr', 'hit', 'hit', 'hitt', 'homicid', 'hostil', 'hurt', 'ingrat', 'injur', 'injury', 'insult', 'invad', 'invas', 'irat', 'irk', 'irritat', 'jealou', 'jealousy', 'jeer', 'kick', 'kil', 'kill', 'knif', 'kniv', 'loath', 'maim', 'mistreat', 'mock', 'murder', 'obliterat', 'offend', 'oppos', 'predatory', 'protest', 'quarrel', 'rage', 'rage', 'raging', 'rapin', 'rebel', 'rebell', 'rebuk', 'relentles', 'reproach', 'resent', 'resentment', 'retribut', 'reveng', 'revolt', 'ridicul', 'rip', 'ripp', 'rip', 'rob', 'robb', 'robs', 'sarcasm', 'sarcastic', 'scalp', 'scof', 'scoff', 'scourg', 'seiz', 'sever', 'severity', 'shatter', 'shoot', 'shot', 'shov', 'slain', 'slander', 'slap', 'slaughter', 'slay', 'slew', 'smash', 'snarl', 'sneer', 'spear', 'spiteful', 'spurn', 'stab', 'steal', 'stol', 'stolen', 'strangl', 'strif', 'strik', 'struck', 'struggl', 'stubborn', 'sword', 'taunt', 'temper', 'threat', 'threaten', 'tore', 'torment', 'torn', 'tortur', 'traitor', 'trampl', 'treacherou', 'treachery', 'tyrant', 'unkind', 'vengeanc', 'vengeful', 'vex', 'vexing', 'violat', 'violenc', 'violent', 'war', 'warring', 'warrior', 'war', 'weapon', 'whip', 'wound', 'wrath', 'football', 'wreck', 'emotionsexpressivebehavior', 'art', 'art', 'bard', 'bark', 'bawl', 'bellow', 'bleat', 'carol', 'chant', 'clown', 'crie', 'criing', 'cry', 'danc', 'exclaim', 'expressiv', 'frisk', 'frolic', 'game', 'guitar', 'harp', 'horn', 'hurrah', 'hurray', 'lullaby', 'lute', 'lute', 'lyre', 'minstrel', 'neigh', 'neigh', 'painter', 'play', 'poem', 'poet', 'poetic', 'poetry', 'roar', 'sang', 'scream', 'shout', 'sigh', 'sing', 'sings', 'sport', 'sung', 'troubador', 'troubadour', 'violin', 'warbl', 'yel', 'yell', 'emotionsglory', 'admir', 'admirabl', 'adventur', 'applaud', 'applaus', 'arroganc', 'arrogant', 'audacity', 'awe', 'boast', 'boastful', 'brillianc', 'brilliant', 'caesar', 'castl', 'conque', 'crown', 'dazzl', 'eagl', 'elit', 'emperor', 'empir', 'exalt', 'exhibit', 'exquisit', 'extraordinary', 'extrem', 'fame', 'famed', 'famou', 'foremost', 'geniu', 'glor', 'gold', 'golden', 'grandeur', 'great', 'haughty', 'hero', 'homag', 'illustriou', 'kingdom', 'magestic', 'magnificent', 'majestic', 'majesty', 'nobl', 'outstand', 'palac', 'pomp', 'prestig', 'prid', 'princ', 'proud', 'renown', 'resplendent', 'rich', 'royal', 'royalty', 'sceptr', 'scorn', 'splendid', 'splendor', 'strut', 'sublim', 'superior', 'superiority', 'suprem', 'thron', 'triump', 'victor', 'victoriou', 'victory', 'wealth', 'wonder', 'wonderful']
primary = ['RidPrimary', 'primaryneed', 'absinth', 'ale', 'alimentary', 'ambrosia', 'ambrosial', 'appetit', 'apple', 'artichok', 'asparagu', 'bacon', 'banana', 'bean', 'beef', 'beer', 'belch', 'belly', 'belly', 'berri', 'berry', 'beverag', 'biscuit', 'bite', 'bite', 'bite', 'biting', 'bitten', 'bonbon', 'brandy', 'bread', 'breakfast', 'breast', 'brew', 'broth', 'burp', 'butter', 'buttermilk', 'cafe', 'cafe', 'cake', 'cake', 'cafetaria', 'candy', 'cannibal', 'caviar', 'champagn', 'chees', 'cherri', 'cherry', 'chestnut', 'chew', 'chok', 'cider', 'claret', 'cob', 'cob', 'cocoa', 'cocoanut', 'coconut', 'coffe', 'consum', 'cook', 'corn', 'cough', 'cranberry', 'cream', 'delicaci', 'delicacy', 'dessert', 'devour', 'diet', 'digest', 'dine', 'dines', 'dining', 'dinner', 'dish', 'dish', 'drank', 'drink', 'drunk', 'drunken', 'eat', 'eaten', 'egg', 'entrail', 'famin', 'famish', 'fast', 'fast', 'fat', 'fatten', 'feast', 'fed', 'feed', 'feed', 'fig', 'fig', 'flour', 'food', 'foodstuff', 'fork', 'fruit', 'garlic', 'gin', 'ginger', 'gin', 'glutton', 'gluttonou', 'gnaw', 'gobbl', 'grain', 'grap', 'grog', 'gruel', 'gulp', 'gum', 'gum', 'gut', 'gut', 'ham', 'ham', 'herb', 'honey', 'hunger', 'hungry', 'imbib', 'inedibl', 'intestin', 'jaw', 'juic', 'lap', 'lap', 'lemon', 'lick', 'lime', 'lime', 'lip', 'lip', 'liqueur', 'liquor', 'lunch', 'maiz', 'meal', 'meat', 'melon', 'menu', 'milk', 'mint', 'morsel', 'mouth', 'mouthful', 'mushroom', 'mutton', 'naus', 'nectar', 'nibbl', 'nourish', 'nourishment', 'nurtur', 'nut', 'nut', 'oliv', 'oral', 'palat', 'partak', 'pastri', 'pastry', 'pea', 'peanut', 'pear', 'pea', 'pepper', 'philtr', 'pineappl', 'poison', 'pork', 'porridg', 'pot', 'potato', 'potbel', 'pot', 'pucker', 'pumpkin', 'quench', 'raspberry', 'raw', 'rawly', 'repast', 'restaurant', 'restaurent', 'rice', 'rice', 'ripenes', 'roast', 'rum', 'rum', 'salad', 'saliva', 'salivat', 'salt', 'sauc', 'sauerkraut', 'sesam', 'sherbert', 'sherry', 'soup', 'spat', 'spit', 'spittl', 'spoon', 'starv', 'starvat', 'stomach', 'strawberri', 'strawberry', 'suck', 'suckl', 'sugar', 'supper', 'swallow', 'tea', 'tea', 'teat', 'teeth', 'thirst', 'thirsty', 'throat', 'tit', 'tit', 'tomato', 'tongu', 'tooth', 'uncook', 'veal', 'vegetabl', 'venison', 'vodka', 'vomit', 'wheat', 'whiskey', 'whisky', 'yam', 'yam', 'yeast', 'anal', 'anus', 'anus', 'arse', 'arsehol', 'ass', 'ass-hol', 'asshol', 'beshat', 'beshit', 'besmear', 'bile', 'bowel', 'buttock', 'cack', 'cesspool', 'cloaca', 'clot', 'clot', 'constipat', 'dank', 'daub', 'defecat', 'defil', 'delous', 'diarrhoea', 'dirt', 'dirty', 'disgust', 'dung', 'dunghill', 'effluvium', 'effluvium', 'enema', 'excret', 'fart', 'fart', 'fecal', 'feces', 'fetid', 'fetor', 'filth', 'filthy', 'impur', 'latrin', 'louse', 'loathsom', 'lous', 'maggot', 'maggoty', 'malodorou', 'malodourou', 'manur', 'mess', 'mess', 'messing', 'miasma', 'mud', 'muddy', 'mud', 'offal', 'ooz', 'oozy', 'outhous', 'piss', 'pollut', 'putrescenc', 'putrescent', 'putrid', 'rancid', 'rectum', 'reek', 'rot', 'rot', 'rotten', 'rotting', 'rump', 'scum', 'sewer', 'shat', 'shit', 'slimy', 'smear', 'sodomist', 'sodomy', 'soil', 'stal', 'stench', 'stink', 'sweat', 'unclean', 'unwash', 'urin', 'venereal', 'adulterer', 'adultery', 'allur', 'bawd', 'bawdy', 'bitch', 'brothel', 'caress', 'carnal', 'circumcis', 'clitori', 'cohabit', 'coitu', 'concubin', 'copulat', 'coquett', 'coquettish', 'courtesan', 'cuckold', 'cunt', 'cupid', 'debauch', 'deflower', 'ejaculat', 'erotic', 'fondl', 'fornicat', 'fuck', 'genital', 'genitalia', 'girdl', 'groin', 'harem', 'harlot', 'homosexual', 'homosexuality', 'immodest', 'incest', 'incestuou', 'indecent', 'indiscret', 'infatuat', 'kiss', 'lasciviou', 'lecher', 'lecherou', 'lechery', 'leer', 'leer', 'lewd', 'libertin', 'licentiou', 'lover', 'lust', 'lustful', 'lusty', 'masturbat', 'menstrual', 'menstruat', 'mistres', 'naked', 'nude', 'nude', 'obscen', 'obscenity', 'orgasm', 'orgi', 'orgy', 'pander', 'paramour', 'peni', 'pervers', 'pervert', 'phallic', 'phallu', 'pregnancy', 'pregnant', 'procreat', 'prostitut', 'prurient', 'puberty', 'pubi', 'pubic', 'rape', 'raping', 'ribald', 'satyr', 'seduc', 'sensual', 'sensuou', 'sex', 'sexed', 'sex', 'sex-linked', 'sexual', 'sexy', 'shameles', 'slattern', 'slut', 'slutty', 'testi', 'testicl', 'thigh', 'trollop', 'unblush', 'undres', 'vagina', 'venu', 'voluptuou', 'vulva', 'waist', 'wanton', 'whor', 'womb', 'primarysensation', 'brush', 'coars', 'contact', 'cudd', 'cuddl', 'handl', 'itch', 'itchy', 'massag', 'prickl', 'rough', 'rub', 'rubb', 'rub', 'scaly', 'scratch', 'sharp', 'slick', 'slippery', 'smooth', 'snuggl', 'sting', 'sting', 'strok', 'textur', 'thick', 'tickl', 'tingl', 'touch', 'waxy', 'aftertast', 'bitter', 'delectabl', 'deliciou', 'flavor', 'gall', 'honi', 'lusciou', 'piquant', 'savor', 'savory', 'savour', 'savoury', 'sour', 'spic', 'spicy', 'sugary', 'sweet', 'sweetnes', 'tang', 'tangy', 'tart', 'tast', 'tasty', 'toothsom', 'unpalatabl', 'unsavory', 'vinegar', 'vinegary', 'aroma', 'aromatic', 'breath', 'cologn', 'fragranc', 'fragrant', 'fume', 'fuming', 'incens', 'inhal', 'musk', 'musky', 'musty', 'nose', 'nostril', 'odor', 'odour', 'perfum', 'pungenc', 'pungent', 'scent', 'smel', 'smell', 'snif', 'sniff', 'apperceive', 'apperceptive', 'attent', 'awar', 'awarenes', 'balmy', 'bask', 'beautiful', 'beauty', 'charm', 'comfort', 'comfortabl', 'creamy', 'fair', 'impress', 'lovelines', 'lush', 'luxuriou', 'luxury', 'milky', 'oversensitiv', 'perceiv', 'percept', 'perceptual', 'physical', 'pleasant', 'pretty', 'refresh', 'relish', 'revel', 'sensat', 'sensitiv', 'stimulat', 'sumptuou', 'auditorilly', 'aloud', 'audibl', 'audition', 'auditory', 'aural', 'bang', 'bell', 'binaural', 'blar', 'boom', 'buzz', 'chord', 'choru', 'clack', 'clamor', 'clamour', 'clang', 'crackl', 'croak', 'deaf', 'dron', 'drum', 'ear', 'ear', 'echo', 'hark', 'hear', 'heard', 'hiss', 'hum', 'humm', 'hum', 'listen', 'loud', 'louder', 'melodi', 'melodiou', 'melody', 'muffl', 'music', 'musical', 'nois', 'noisy', 'peal', 'purr', 'racket', 'rasp', 'rattl', 'raucou', 'resonant', 'resound', 'rhythm', 'ring', 'rumbl', 'rustl', 'serenad', 'shrill', 'snap', 'sonorou', 'sound', 'stridant', 'strident', 'swish', 'symphony', 'tempo', 'thud', 'timbr', 'tinkl', 'tonal', 'tone', 'toned', 'tone', 'trill', 'tune', 'tuned', 'tune', 'tuning', 'unheard', 'vocal', 'voic', 'whir', 'whirr', 'whistl', 'after-image', 'blink', 'illuminant', 'invisibility', 'monocular', 'amber', 'appear', 'appearanc', 'aurora', 'azur', 'beam', 'behold', 'binocular', 'blue', 'bluish', 'bright', 'brown', 'brunett', 'chromatic', 'color', 'colour', 'complex', 'crimson', 'discern', 'dye', 'emerald', 'film', 'flash', 'flicker', 'flourescent', 'gaze', 'gazing', 'glanc', 'glar', 'gleam', 'glimps', 'glint', 'glisten', 'glitter', 'glossy', 'glow', 'gray', 'green', 'grey', 'halo', 'hue', 'illuminat', 'imag', 'invisibl', 'lamp', 'lantern', 'lavender', 'light', 'lighten', 'lightn', 'limpid', 'look', 'lucid', 'luminance', 'luminou', 'luster', 'lustrou', 'moonbeam', 'moonlight', 'notic', 'observ', 'opaqu', 'paint', 'peek', 'peer', 'pictur', 'pink', 'radianc', 'radiant', 'ray', 'ray', 'regard', 'rosy', 'roug', 'ruby', 'ruddy', 'sapphir', 'saw', 'scan', 'scann', 'scan', 'scarlet', 'scen', 'scenic', 'see', 'seeing', 'seen', 'see', 'sheen', 'shimmer', 'shin', 'shon', 'sight', 'sparkl', 'spied', 'spy', 'spy', 'spying', 'star', 'starlight', 'star', 'sunlight', 'sunshin', 'survey', 'tan', 'tanned', 'tanning', 'tan', 'tint', 'translucent', 'transparent', 'twinkl', 'unseen', 'view', 'violet', 'visibl', 'vision', 'visual', 'watch', 'witnes', 'yellow', 'alaska', 'arctic', 'benumb', 'chil', 'chill', 'cold', 'colder', 'cool', 'freez', 'frigid', 'frost', 'frostbit', 'froz', 'frozen', 'glacier', 'hoar', 'ice', 'icines', 'icing', 'icy', 'north', 'northern', 'numb', 'numbness', 'polar', 'shiver', 'siberia', 'sleet', 'snow', 'snowstorm', 'snowy', 'thul', 'winter', 'wintry', 'alabaster', 'bra', 'brassy', 'brazen', 'brittl', 'bronz', 'copper', 'crisp', 'crispy', 'glas', 'glassy', 'granit', 'gravel', 'hard', 'iron', 'marbl', 'metal', 'metallic', 'nail', 'pebb', 'porcelain', 'rigid', 'rock', 'solid', 'splinter', 'steel', 'stiff', 'ston', 'stony', 'zinc', 'damask', 'delicat', 'downy', 'feather', 'fleec', 'fleecy', 'fluffy', 'gentl', 'gentlenes', 'gossamer', 'lace', 'lace', 'lacing', 'lacy', 'mellow', 'mild', 'murmur', 'pliant', 'powdery', 'satin', 'satiny', 'silk', 'soft', 'tender', 'ting', 'velvet', 'velvety', 'whisper', 'primarydefensivesymbol', 'stagnant', 'apathetic', 'apathy', 'bed', 'bedd', 'bed', 'boredom', 'calm', 'contented', 'contentment', 'couch', 'cozy', 'dead', 'death', 'decay', 'die', 'died', 'dy', 'dormant', 'drift', 'dying', 'ease', 'eased', 'eas', 'hush', 'idl', 'immobil', 'inactiv', 'inactivity', 'indifferenc', 'indifferent', 'indolent', 'inert', 'inertia', 'innert', 'laid', 'lain', 'langorou', 'languid', 'languish', 'languor', 'lassitud', 'lay', 'laying', 'lay', 'lazy', 'leaden', 'leisur', 'lethargic', 'lethargy', 'lie', 'lie', 'linger', 'listles', 'lul', 'lull', 'motionles', 'nestl', 'nonchalanc', 'nonchalant', 'passiv', 'passivity', 'peaceful', 'perish', 'phlegmatic', 'placid', 'procrastinat', 'quiet', 'relax', 'relaxat', 'repos', 'rest', 'restful', 'retir', 'safe', 'safely', 'safety', 'secur', 'security', 'sedentary', 'seren', 'serenity', 'silenc', 'silent', 'slack', 'slothful', 'slow', 'sluggish', 'solac', 'sooth', 'stagnat', 'static', 'stillnes', 'submiss', 'submissiv', 'submit', 'succumb', 'tranq', 'unhurri', 'vagrant', 'velleity', 'wearisom', 'weary', 'yield', 'caravan', 'chas', 'cruis', 'desert', 'driv', 'embark', 'emigrat', 'explor', 'immigrat', 'immigrant', 'journey', 'migrat', 'navigat', 'nomad', 'nomadic', 'oscillat', 'pilgrim', 'pilgrimag', 'ride', 'ride', 'riding', 'roam', 'rode', 'rov', 'sail', 'sailor', 'seafar', 'search', 'ship', 'stray', 'tour', 'tourist', 'travel', 'trek', 'trip', 'vagabond', 'voyag', 'wander', 'wanderlust', 'wayfarer', 'wildernes', 'yonder', 'activiti', 'activity', 'agitat', 'churn', 'commot', 'convuls', 'expand', 'expans', 'fidget', 'flounder', 'flurri', 'flurry', 'jerk', 'lurch', 'orbit', 'pitch', 'pivot', 'pul', 'pulsat', 'quak', 'quiver', 'reel', 'revolv', 'rol', 'roll', 'rotat', 'seeth', 'shak', 'shook', 'spasm', 'spin', 'spread', 'stagger', 'stir', 'sway', 'swel', 'swell', 'swivel', 'swollen', 'throb', 'totter', 'twich', 'twist', 'twitch', 'undulat', 'vibrat', 'wave', 'waved', 'wave', 'waving', 'whirl', 'wobbl', 'blur', 'cloud', 'cloudy', 'curtain', 'darken', 'diffus', 'dim', 'dimm', 'dims', 'equivocal', 'fade', 'faded', 'fade', 'fading', 'fog', 'fogg', 'fog', 'haze', 'hazing', 'hazy', 'indefinit', 'indistinct', 'mist', 'misty', 'murkines', 'murky', 'nebula', 'nebulou', 'obscur', 'overcast', 'screen', 'shad', 'shadow', 'shadowy', 'shady', 'twilight', 'uncertain', 'uncertaint', 'unclear', 'vagu', 'vapor', 'vapour', 'veil', 'aimles', 'ambiguit', 'ambiguou', 'anarchy', 'chanc', 'chao', 'char', 'char', 'catastrophe', 'confus', 'crowd', 'discord', 'discordant', 'dishevel', 'disorder', 'entangl', 'gordian', 'haphazard', 'irregular', 'jumbl', 'jungl', 'labyrinth', 'lawles', 'litter', 'mob', 'mobb', 'mob', 'overgrown', 'overrun', 'perplex', 'random', 'ruin', 'unru', 'wild', 'primaryregressivecognition', 'bizzar', 'bodiles', 'boundles', 'cryptic', 'enigma', 'esoteric', 'exotic', 'fantastic', 'formles', 'immeasurabl', 'inconceivabl', 'incredibl', 'indescribabl', 'ineffabl', 'infinity', 'inscrutabl', 'limitles', 'magi', 'magic', 'magu', 'marvel', 'myst', 'nameles', 'nothingnes', 'numberles', 'occult', 'odd', 'secrecy', 'secret', 'shapeles', 'sorcerer', 'sorceres', 'strang', 'transcend', 'unbelievabl', 'unbound', 'unimaginabl', 'unknown', 'unlimit', 'unspeakabl', 'untold', 'void', 'aeon', 'ceaseles', 'centuri', 'century', 'continual', 'continuou', 'endles', 'endur', 'eon', 'eternal', 'eternity', 'everlast', 'forever', 'immortal', 'incessant', 'lifetim', 'outliv', 'permanenc', 'permanent', 'perpetual', 'timelessnes', 'unceas', 'undy', 'unend', 'test5', 'amuck', 'asleep', 'awak', 'awaken', 'bedlam', 'coma', 'craz', 'crazy', 'deliriou', 'delirium', 'delphic', 'dement', 'doze', 'dozed', 'doze', 'dozing', 'dream', 'dreamy', 'drowsy', 'ecstacy', 'ecstasy', 'ecstatic', 'enchant', 'epilepsy', 'epileptic', 'exstasy', 'faint', 'fantasi', 'fantasy', 'febril', 'fever', 'feverish', 'frenzy', 'hallucinat', 'hashish', 'hibernat', 'hypno', 'hysteria', 'hysteric', 'imagin', 'imaginat', 'insan', 'insanity', 'intuit', 'irrational', 'laudanum', 'lunacy', 'lunatic', 'mad', 'madly', 'madman', 'madman', 'madnes', 'madwoman', 'madwoman', 'mania', 'maniac', 'meditat', 'mesmeriz', 'monomania', 'nap', 'napp', 'nap', 'neurosi', 'neurotic', 'nightmar', 'nightmarish', 'opium', 'opiate', 'oracl', 'parano', 'premonit', 'psychic', 'psychosi', 'psychotic', 'raptur', 'rapturou', 'reveri', 'revery', 'reviv', 'seer', 'sleep', 'sleepy', 'slumber', 'stupor', 'swoon', 'telepathy', 'tranc', 'unreason', 'vertigo', 'visionary', 'wak', 'woke', 'acces', 'aisl', 'aqueduct', 'arteri', 'artery', 'avenu', 'barrier', 'border', 'boundari', 'boundary', 'bridg', 'brim', 'brink', 'canal', 'channel', 'coast', 'conduit', 'corridor', 'curb', 'door', 'doorstep', 'doorway', 'edg', 'entranc', 'entry', 'fenc', 'ferri', 'ferry', 'floor', 'footpath', 'foyer', 'fram', 'fring', 'frontier', 'gate', 'gating', 'hall', 'hallway', 'highway', 'horizon', 'lane', 'lane', 'ledg', 'line', 'lined', 'line', 'lining', 'margin', 'passag', 'passageway', 'path', 'perimet', 'peripher', 'port', 'railroad', 'railway', 'rim', 'rimm', 'rim', 'road', 'rout', 'sidewalk', 'skylin', 'stair', 'step', 'street', 'threshold', 'trail', 'verg', 'viaduct', 'vista', 'wall', 'window', 'arm', 'arm', 'beard', 'blood', 'bodi', 'body', 'bone', 'bone', 'brain', 'brow', 'brow', 'cheek', 'chest', 'chin', 'corp', 'eye', 'face', 'face', 'facies', 'foot', 'flesh', 'foot', 'forehead', 'hair', 'hand', 'head', 'heart', 'heel', 'hip', 'hip', 'kidney', 'knee', 'knee', 'leg', 'leg', 'limb', 'liver', 'muscl', 'navel', 'neck', 'organ', 'palm', 'rib', 'rib', 'shoulder', 'skin', 'skull', 'thumb', 'toe', 'toe', 'vein', 'wrist', 'acros', 'afar', 'afield', 'ahead', 'along', 'among', 'apart', 'asid', 'at', 'away', 'back', 'behind', 'besid', 'between', 'center', 'centr', 'circl', 'clos', 'closer', 'corner', 'curv', 'distanc', 'distant', 'east', 'eastern', 'everywher', 'extend', 'extensiv', 'extent', 'far', 'farther', 'flat', 'forward', 'front', 'further', 'here', 'hither', 'insid', 'interior', 'layer', 'length', 'level', 'long', 'middl', 'midst', 'narrow', 'near', 'nearby', 'nearer', 'nearest', 'off', 'open', 'out', 'outing', 'out', 'outsid', 'outward', 'over', 'plac', 'point', 'posit', 'rear', 'region', 'round', 'separat', 'side', 'sided', 'side', 'siding', 'situat', 'somewher', 'south', 'spac', 'spaciou', 'spatial', 'squar', 'straight', 'surfac', 'surround', 'thenc', 'thither', 'tip', 'tipp', 'tip', 'toward', 'west', 'western', 'wher', 'wherever', 'wide', 'width', 'within', 'primaryicarianimagery', 'aloft', 'aris', 'arisen', 'aros', 'ascend', 'ascens', 'bounc', 'climb', 'dangl', 'dawn', 'flap', 'fled', 'flew', 'flier', 'flight', 'fling', 'float', 'flown', 'flung', 'flutter', 'fly', 'hang', 'hover', 'hurl', 'icarian', 'icaru', 'jump', 'leap', 'lept', 'lift', 'mount', 'mountainsid', 'rise', 'risen', 'rise', 'rising', 'soar', 'sprang', 'spring', 'sprung', 'sunris', 'swing', 'threw', 'throw', 'thrown', 'toss', 'uphill', 'upward', 'wing', 'abov', 'aerial', 'airplan', 'arch', 'atmospher', 'balcony', 'battlement', 'bird', 'branch', 'ceil', 'cliff', 'crag', 'craggy', 'dome', 'dome', 'doming', 'elevat', 'erect', 'grew', 'grow', 'grown', 'heap', 'heaven', 'height', 'high', 'higher', 'hill', 'hillsid', 'hilltop', 'hung', 'ladder', 'loft', 'lofty', 'mound', 'mountain', 'obelisk', 'overhead', 'peak', 'pile', 'piling', 'planet', 'precipic', 'pyramid', 'rafter', 'rainbow', 'rampart', 'ridg', 'roof', 'sky', 'slop', 'spir', 'steep', 'summit', 'tall', 'taller', 'tallest', 'top', 'topp', 'top', 'tower', 'tree', 'trelli', 'upper', 'uppermost', 'zenith', 'base', 'base', 'buri', 'burrow', 'bury', 'descend', 'descent', 'dig', 'digg', 'dig', 'dip', 'dipp', 'dip', 'dive', 'downhill', 'downstream', 'droop', 'drop', 'drop', 'dug', 'fall', 'fallen', 'fell', 'headlong', 'lean', 'plung', 'reced', 'reclin', 'sank', 'sink', 'slid', 'slip', 'stoop', 'sundown', 'sunk', 'sunken', 'sunset', 'swoop', 'toppl', 'tumbl', 'below', 'beneath', 'bottom', 'canyon', 'cave', 'caving', 'cellar', 'chasm', 'crevas', 'deep', 'deeper', 'depth', 'ditch', 'downward', 'gutter', 'hole', 'hole', 'low', 'pit', 'pit', 'pitt', 'precipitou', 'ravin', 'root', 'submarin', 'trench', 'tunnel', 'under', 'underground', 'underneath', 'underworld', 'valley', 'solar', 'ablaz', 'afir', 'ash', 'ash', 'blast', 'blaz', 'boil', 'broil', 'burn', 'burnt', 'candl', 'charcoal', 'coal', 'combust', 'ember', 'fiery', 'fire', 'flam', 'hearth', 'heat', 'hot', 'ignit', 'inferno', 'inflam', 'kindl', 'lit', 'melt', 'scorch', 'sear', 'sizzl', 'smok', 'smolder', 'smoulder', 'spark', 'sultry', 'sun', 'sunn', 'sun', 'sunstrok', 'tropic', 'tropical', 'warm', 'warmth', 'bath', 'beach', 'brook', 'bubbl', 'bucket', 'creek', 'dam', 'damm', 'damp', 'dam', 'dew', 'dew', 'dewy', 'dike', 'downpour', 'drench', 'shoring', 'surf', 'surfing', 'drip', 'fen', 'flood', 'fluid', 'foam', 'fountain', 'gurgl', 'humid', 'lake', 'lake', 'liquid', 'moat', 'moist', 'moistur', 'moss', 'moss', 'ocean', 'overflow', 'perspir', 'perspirat', 'pond', 'pool', 'pour', 'rain', 'rainfall', 'river', 'saturat', 'sea', 'sea', 'shore', 'shore', 'shower', 'soak', 'splash', 'sprinkl', 'steam', 'steamy', 'stream', 'swam', 'swamp', 'swampy', 'swim', 'swum', 'tide', 'tide', 'tiding', 'trickl', 'wade', 'wading', 'wash', 'water', 'waterfall', 'wet']
secondary = ['RidSecondary', 'secondaryabstraction', 'diverse', 'diversification', 'diversified', 'diversity', 'evident', 'evidential', 'guess', 'logistic', 'abstract', 'almost', 'alternativ', 'analy', 'attribut', 'axiom', 'basic', 'belief', 'believ', 'calculat', 'caus', 'certain', 'characteriz', 'choic', 'choos', 'chos', 'circumstanc', 'comprehend', 'compar', 'comprehens', 'conditional', 'concentrat', 'concept', 'conclud', 'conjectur', 'consequenc', 'consequent', 'consider', 'contriv', 'criter', 'criterion', 'decid', 'deem', 'defin', 'deliberat', 'determin', 'differenc', 'different', 'distinct', 'distinguish', 'doctrin', 'effect', 'establish', 'estimat', 'evaluat', 'evidenc', 'examin', 'exampl', 'except', 'fact', 'fact', 'featur', 'figur', 'forethought', 'formulat', 'gues', 'history', 'idea', 'importanc', 'important', 'informat', 'interpret', 'interpretat', 'judg', 'judgment', 'knew', 'know', 'learn', 'logic', 'may', 'meant', 'mistak', 'mistaken', 'mistook', 'model', 'opin', 'otherwis', 'perhap', 'plan', 'possi', 'predicat', 'predict', 'probab', 'probabl', 'problem', 'proof', 'prov', 'purpos', 'quali', 'quant', 're-analy', 're-examin', 'rational', 'real', 'reality', 'reason', 'reasonabl', 'reconsider', 'reexamin', 'reformulat', 'reinterpretat', 'relearn', 'relevanc', 'relevant', 'research', 'resolv', 'schem', 'scienc', 'scientific', 'select', 'significanc', 'solut', 'someth', 'somewhat', 'sourc', 'subject', 'suppos', 'sure', 'surely', 'tend', 'them', 'theor', 'think', 'thinker', 'thought', 'topic', 'true', 'truly', 'truth', 'ttt1', 'understand', 'understood', 'weigh', 'weighed', 'weighing', 'weighs', 'why', 'secondarysocialbehaviour', 'guest', 'quota', 'quota-', 'quota', 'acquiescence', 'approbation', 'consensus', 'consult', 'prosocial', 'sociable', 'able', 'accept', 'acceptanc', 'addres', 'admit', 'advic', 'advis', 'agre', 'aid', 'allow', 'announc', 'answer', 'apologis', 'apologiz', 'appeal', 'approv', 'approval', 'ask', 'asked', 'asking', 'asks', 'assist', 'assur', 'bargain', 'beckon', 'beseech', 'borrow', 'call', 'comment', 'commit', 'communicat', 'conduct', 'confer', 'confes', 'confid', 'confirm', 'congratulat', 'consent', 'consol', 'consolat', 'convers', 'conversat', 'convinc', 'cooperat', 'counsel', 'declar', 'depend', 'dependent', 'describ', 'dialogu', 'discours', 'discus', 'discus', 'donat', 'educat', 'elect', 'encourag', 'encouragement', 'engag', 'escort', 'excus', 'explain', 'follow', 'forgav', 'forgiv', 'forgiven', 'generosity', 'generou', 'gift', 'grant', 'greet', 'guid', 'guidanc', 'help', 'imitat', 'implor', 'influenc', 'inform', 'inquir', 'instruct', 'interview', 'introduc', 'invit', 'kneel', 'lend', 'lent', 'meet', 'ment', 'messag', 'met', 'mutual', 'offer', 'pardon', 'participat', 'persuad', 'persua', 'plead', 'plea', 'preach', 'proclaim', 'promis', 'propos', 'protect', 'provid', 'quot', 'recit', 'reeducation', 'remark', 'remind', 'repli', 'reply', 'represent', 'request', 'rescu', 'respond', 'respons', 'said', 'sale', 'sale', 'say', 'servic', 'shar', 'shelter', 'signal', 'social', 'solicit', 'speak', 'speaker', 'speech', 'spok', 'spoken', 'suggest', 'sworn', 'talk', 'taught', 'teach', 'tell', 'thank', 'told', 'treat', 'utter', 'visit', 'secondaryinstrumentalbehavior', 'avail', 'caveat', 'divestment', 'dividend', 'foundr', 'laborator', 'spin-off', 'availability', 'component', 'ingredient', 'logistics', 'merchandise', 'provision', 'achiev', 'achievement', 'acquir', 'acquisit', 'afford', 'aim', 'applic', 'applie', 'apply', 'architect', 'assembl', 'attain', 'attempt', 'availabl', 'belong', 'bid', 'bought', 'build', 'built', 'burden', 'busines', 'buy', 'capabl', 'carri', 'carry', 'claim', 'collect', 'construct', 'copi', 'copy', 'cost', 'count', 'craft', 'craftsman', 'cultivat', 'cure', 'curing', 'deliver', 'earn', 'effort', 'employ', 'endeavor', 'factori', 'factory', 'feat', 'feat', 'find', 'finish', 'forge', 'forge', 'found', 'gain', 'goal', 'grasp', 'harvest', 'hire', 'hired', 'hire', 'hiring', 'improv', 'industri', 'industry', 'job', 'job', 'labor', 'laboriou', 'labour', 'labouriou', 'lesson', 'machin', 'machinery', 'mak', 'manipulat', 'manufactur', 'market', 'mend', 'merchant', 'money', 'obtain', 'occupat', 'occupy', 'ownership', 'paid', 'pay', 'paying', 'pay', 'perform', 'pick', 'plough', 'plow', 'posse', 'posse', 'practic', 'prepar', 'pric', 'privation', 'produc', 'profit', 'profitabl', 'property', 'purchas', 'pursu', 'reach', 'reconstruct', 'record', 'recover', 'repair', 'reproduce', 'restor', 'result', 'risk', 'sel', 'sell', 'skil', 'skill', 'skillful', 'sold', 'sow', 'spend', 'spent', 'student', 'studi', 'studiou', 'study', 'succe', 'sweep', 'swept', 'task', 'test', 'toil', 'toiled', 'toil', 'trad', 'tried', 'try', 'trying', 'try', 'use', 'used', 'us', 'using', 'win', 'winning', 'win', 'won', 'work', 'secondaryrestraint', 'comptroller', 'discipline', 'magist', 'penaliz', 'penitentiary', 'arrest', 'assign', 'authoriz', 'bar', 'barred', 'barring', 'bar', 'bind', 'block', 'blockad', 'bound', 'cag', 'captiv', 'captivity', 'captur', 'catch', 'caught', 'censur', 'chastis', 'chastiz', 'coerc', 'compel', 'confin', 'conform', 'conformity', 'constrain', 'constraint', 'constrict', 'control', 'decree', 'detain', 'deter', 'dungeon', 'enclos', 'forbad', 'forbid', 'forbidden', 'guard', 'guardian', 'halt', 'hamper', 'hinder', 'hindranc', 'imperativ', 'imprison', 'inhibit', 'insist', 'interfer', 'interrupt', 'jail', 'leash', 'limit', 'lock', 'manag', 'must', 'necessary', 'necessity', 'obedienc', 'obey', 'oblig', 'obligat', 'obstacl', 'obstruct', 'penalti', 'penalty', 'permiss', 'permit', 'polic', 'policeman', 'policeman', 'prescrib', 'prevail', 'prevent', 'prison', 'prohibit', 'punish', 'punishment', 'refus', 'regulat', 'reign', 'requir', 'requirement', 'resist', 'restrain', 'restraint', 'restrict', 'scold', 'shut', 'stop', 'strict', 'summon', 'suppres', 'taboo', 'tax', 'thwart', 'secondaryorder', 'ordinal', 'accurat', 'arrang', 'array', 'balanc', 'catalog', 'class', 'consistenc', 'consistent', 'constanc', 'constant', 'divid', 'form', 'formula', 'grad', 'index', 'list', 'measur', 'method', 'moderat', 'neat', 'norm', 'normal', 'organi', 'order', 'pattern', 'precis', 'rank', 'regular', 'reorganiz', 'routin', 'serial', 'series', 'simpl', 'simplicity', 'stability', 'standard', 'symmetr', 'system', 'uniform', 'universal', 'secondarytemporalreference', 'full-time', 'long-term', 'longevit', 'part-time', 'short-term', 'abrupt', 'again', 'ago', 'already', 'ancient', 'brevity', 'brief', 'clock', 'daily', 'date', 'dated', 'date', 'dating', 'decad', 'dur', 'durat', 'earlier', 'early', 'ephemeral', 'ever', 'former', 'frequent', 'hast', 'henceforth', 'hour', 'immediat', 'immediate', 'instant', 'interlud', 'meantim', 'meanwhil', 'minut', 'moment', 'momentary', 'month', 'now', 'occas', 'occasional', 'often', 'old', 'older', 'once', 'past', 'prematur', 'present', 'previou', 'prior', 'quick', 'season', 'seldom', 'sometim', 'soon', 'sooner', 'sudden', 'temporary', 'then', 'till', 'time', 'timing', 'today', 'tonight', 'week', 'when', 'whenever', 'whil', 'year', 'yesterday', 'secondarymoralimperative', 'legitimacy', 'respect', 'birthright', 'commandment', 'conscienc', 'conscientiou', 'correct', 'custom', 'customer', 'customiz', 'duti', 'duty', 'ethic', 'honest', 'honesty', 'honor', 'honorabl', 'honour', 'honourabl', 'justic', 'law', 'lawful', 'law', 'legal', 'legitimat', 'moral', 'morality', 'ought', 'prerogativ', 'principl', 'privileg', 'proper', 'rectitud', 'respectful', 'responsibility', 'responsibl', 'right', 'righteou', 'rightful', 'sanct', 'should', 'trustworthy', 'unjust', 'upright', 'virtu']

# Function to replace word in text by corresponding dictionary category name
def swap(string,dic,rep):
        pattern = re.compile(r"\b(" + "|".join(dic) + r")\b")
        filtered = pattern.sub(rep, string)
        return filtered

# Function to replace unreplaced words by "x" to further isolate RID feature
def resttox2 (string):
        p = re.compile(r'(?!RidPrimary|RidSecondary|RidEmotions)\b\w+\b')
        filtered = p.sub('x', string)
        return filtered

# Creating the RID tags and add to dataframe
df['RidTag'] = df['CleanLyrics'].apply(lambda text: swap(text,primary[1:],primary[0]))
df['RidTag'] = df['RidTag'].apply(lambda text: swap(text,secondary[1:],secondary[0]))
df['RidTag'] = df['RidTag'].apply(lambda text: swap(text,emotions[1:],emotions[0]))

# Swap the unreplaced words by "x" and add to dataframe
df['RidTagOnly'] = df['RidTag'].apply(resttox2)

print(df.head(3))


# In[130]:


from nltk.stem.porter import PorterStemmer
from itertools import groupby

#Extract lexical features: Vocabulary Richness, 
# Average word per line, Average word lenght, Egotest

# Create feature for Average word lenght
def wordlenght (string):
    words = string.split()
    wordlen = sum(len(word) for word in words)
    numwords = len(words)
    average = wordlen/numwords
    return average 

df['WordLenght'] = df.CleanLyrics.apply(wordlenght)


# Create feature for Average number of word per line
def wordperline(entry):
    wordcount = len(entry.split())
    linecount = len(entry.split('\n'))
    wordline = (wordcount/linecount)
    return wordline
    
df['WordPerLine'] = df.Lyrics.apply(wordperline)
    
    
# Create functions for Vocabulary Richness based on Yule's Inverse Rule
def words(entry):
    words = filter(lambda w: len(w) > 0, [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])
    return words
def yule(entry):
    d = {}
    stemmer = PorterStemmer()
    for w in words(entry):
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1

    M1 = float(len(d))
    M2 = sum([len(list(g)) * (freq ** 2) for freq, g in
              groupby(sorted(d.values()))])

    try:
        val = (M1 * M1) / (M2 - M1)
        return val
    except ZeroDivisionError:
        return 0

df['VocabularyRichness'] = df.CleanLyrics.apply(yule)


# Create feature for egotest
#Count up different personal words in each song
pw1 = df.CleanLyrics.str.count(' me ')
pw2 = df.CleanLyrics.str.count(' my ')
pw3 = df.CleanLyrics.str.count(' mine ')
pw4 = df.CleanLyrics.str.count(' myself ')
pw5 = df.CleanLyrics.str.count(' i ')
X = pw1+pw2+pw3+pw4+pw5 # Total personal words in each song
Y = df.CleanLyrics.str.len() # Total words in each song

df['Egotest'] = X/Y # Personal words / Total words Log scale

print(df.head(3))


# In[148]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import svm

# Grid search Bag-of-Words vectorization parameters

svmcl = svm.SVC()

# Create Pipeline
pipeline = Pipeline([
    ('BOW', CountVectorizer(analyzer='word')),
    ('svmcl', svm.SVC()),
])

# Set parameters to search
parameters = {
    'BOW__ngram_range': ((1,1),(2,2),(3,3), (4,4),(5,5)),
    'BOW__max_features': (10, 25, 50, 100,250),
}

# Perform grid search
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                           verbose=1, scoring = "accuracy", 
                           refit=True, cv=10)

# Print results of grid search
print ("Performing grid search...")
print ("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(df.CleanLyrics, df.Artist) 
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))  


# In[149]:


#Pipeline to grid search Part-of-Speech Tags vectorization parameters

pipeline = Pipeline([
    ('POS', CountVectorizer(analyzer='word')),
    ('svmcl', svm.SVC()),
])

parameters = {
    'POS__ngram_range': ((1,1),(2,2),(3,3), (4,4),(5,5)),
    'POS__max_features': (10, 25, 50, 100,250),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                           verbose=1, scoring = "accuracy", 
                           refit=True, cv=10)

print ("Performing grid search...")
print ("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(df.PosTag, df.Artist) 
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))  


# In[150]:


#Pipeline to grid search Character vectorization tuning parameters

pipeline = Pipeline([
    ('POSWORD', CountVectorizer(analyzer='word')),
    ('svmcl', svm.SVC()),
])

parameters = {
    'POSWORD__ngram_range': ((1,1),(2,2),(3,3), (4,4),(5,5)),
    'POSWORD__max_features': (10, 25, 50, 100,250),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                           verbose=1, scoring = "accuracy", 
                           refit=True, cv=10)

print ("Performing grid search...")
print ("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(df.PosWord, df.Artist) 
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[153]:


#Pipeline to grid search RID tags vectorization tuning parameters

pipeline = Pipeline([
    ('RID', CountVectorizer(analyzer='word')),
    ('svmcl', svm.SVC()),
])

parameters = {
    'RID__ngram_range': ((1,1),(2,2),(3,3), (4,4),(5,5)),
    'RID__max_features': (10, 25, 50, 100,250),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                           verbose=1, scoring = "accuracy", 
                           refit=True, cv=10)

print ("Performing grid search...")
print ("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(df.RidTag, df.Artist) 
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))  


# In[154]:


#Pipeline to grid search RID Only tags vectorization tuning parameters

pipeline = Pipeline([
    ('RIDONLY', CountVectorizer(analyzer='word')),
    ('svmcl', svm.SVC()),
])

parameters = {
    'RIDONLY__ngram_range': ((1,1),(2,2),(3,3), (4,4),(5,5)),
    'RIDONLY__max_features': (10, 25, 50, 100,250),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                           verbose=1, scoring = "accuracy", 
                           refit=True, cv=10)

print ("Performing grid search...")
print ("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(df.RidTagOnly, df.Artist) 
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))   


# In[159]:


# All features combined via Mapper (modified according to feature vector investigated)

from sklearn_pandas import DataFrameMapper, cross_validation

features = DataFrameMapper([
    (['VocabularyRichness','Egotest','WordPerLine','WordLenght'], None),
    ('CleanLyrics',CountVectorizer(analyzer = "word",   \
                             ngram_range=(1, 1),    \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 250)),
    ('PosTag',CountVectorizer(analyzer = "word",   \
                             ngram_range=(2, 2),    \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 250)),
    ('PosWord',CountVectorizer(analyzer = "word",   \
                             ngram_range=(1, 1),    \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 250)),
    ('RidTag',CountVectorizer(analyzer = "word",   \
                             ngram_range=(2, 2),    \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 250)),
    ('RidTagOnly',CountVectorizer(analyzer = "word",   \
                             ngram_range=(4, 4),    \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 25))])


# In[164]:


#SVM

# Transform features dataframe accordingly 
train = features.fit_transform(df)
train = pd.DataFrame(train)

# Set Parameters to grid search
parameters = {'C': [0.001, 0.01, 0.1, 1, 10],
              'kernel': ['linear','rbf'],
              'gamma' : [0.001, 0.01, 0.1, 1]
              }

#Grid search
grid_search = GridSearchCV(svmcl, parameters, n_jobs=-1, cv=10)
grid_search.fit(train, df.Artist)

# Print Results
print ("Best score: %0.3f" % grid_search.best_score_)
print ("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))

