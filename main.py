import transformers
import torch
import torch.utils.data as datautils
import torch.nn as nn
from IPython import embed
import argparse
import transformers
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, BasicTokenizer
import pickle
import csv
import re
import numpy as np
import os
import tqdm
from transformers import XLNetConfig
from transformers.optimization import AdamW

class NaiveTokenizer(object):
    def __init__(self, from_pretrained=None):
        fin = None
        self.str2idx = {" " : 0, "<s>" : 1, "</s>" : 2, "<sep>": 3, "<mask>": 4, "<pad>": 5}
        self.idx2str = ["", "<s>", "</s>", "<sep>", "<mask>", "<pad>"]
        if from_pretrained is not None:
            try:
                fin = open(from_pretrained, "rb")
            except:
                fin = None

            if fin is not None:
                self.str2idx, self.idx2str = pickle.load(fin)
                fin.close()
            else:
                print("Warning: pretrained file at location \"%s\" not found." % from_pretrained)

        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, never_split=["<mask>", "</s>", "<EOT>", "<EOL>", "<sep>", "<mask>", "<pad>"])
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.pad_token_id = 5
    def dump(self, path):
        with open(path, "wb") as fout:
            pickle.dump((self.str2idx, self.idx2str), fout)

    def tokenize(self, string, max_len=None):

        text = re.sub(r"\x89Û_", "", string)
        text = re.sub(r"\x89ÛÒ", "", text)
        text = re.sub(r"\x89ÛÓ", "", text).lower
        tokens = self.basic_tokenizer.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in self.str2idx:
                token_ids.append(self.str2idx[token])
            else:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)
                token_ids.append(self.str2idx[token])
        if max_len is not None:
            while len(token_ids) < max_len:
                token_ids.append(1)
        return token_ids

    def encode(self, string, max_len=None):
        tokens = self.basic_tokenizer.tokenize(string)
        token_ids = []
        for token in tokens:
            if token in self.str2idx:
                token_ids.append(self.str2idx[token])
            else:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)
                token_ids.append(self.str2idx[token])
        if max_len is not None:
            while len(token_ids) < max_len:
                token_ids.append(1)
        return token_ids

    def decode(self, list_of_idx):
        if type(list_of_idx) is int:
            list_of_idx = [list_of_idx]
        ret = []
        for idx in list_of_idx:
            if idx < self.size():
                ret.append(self.idx2str[idx])
            else:
                ret.append("UNKTOKEN")
        return " ".join(ret)

    def size(self):
        return len(self.idx2str)

    def __len__(self):
        return len(self.idx2str)

### Credits to https://www.kaggle.com/arijzou/text-preprocessing-disaster-tweets

def remove_urls (text):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', text)
    return text
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return text


from emot.emo_unicode import UNICODE_EMO, EMOTICONS
def remove_emojis(text):
    sentence = text.split()
    new_sentence = []
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    # since we have an emoticon ":/" as substing in any url, we need to prevent replacing it
    url_keep_pattern = re.compile("https?://")

    for w in sentence:
        w = emoji_pattern.sub(r'', w)
        if (url_keep_pattern.match(w) is None):  # if it's not a url
            w = emoticon_pattern.sub(r'', w)
        new_sentence.append(w)

    return (" ".join(new_sentence))
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "after midday",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}
def decontracted(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text
def fix_slangs(text):
    # first of all we need to change our text to lowercase to match the abbreviations, otherwise some words won't be changed
    # eg: NYC
    text = text.lower()
    sentence_list = text.split()
    new_sentence = []

    for word in sentence_list:
        for candidate_replacement in abbreviations:
            if (candidate_replacement == word):
                word = word.replace(candidate_replacement, abbreviations[candidate_replacement])
        new_sentence.append(word)

    return (" ".join(new_sentence))

### End credits

class DisasterTweetsClassificationDataset(datautils.Dataset):
    def __init__(self, tokenizer, split_path: str, split_type: str):
        super(DisasterTweetsClassificationDataset, self).__init__()
        self.is_training = split_type.lower() == "train"
        self.is_dev = False
        self.data_pool = torch.zeros((15000, 100), dtype=torch.long)
        self.mask_pool = torch.zeros((15000, 100), dtype=torch.long)
        self.label_pool = torch.zeros((15000,), dtype=torch.long)
        self.training_indices = []
        self.dev_indices = []

        with open(split_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row["id"])
                sequence = "# %s # %s # %s" % (row["keyword"], row["location"], row["text"])
                sequence = remove_emojis(sequence)
                sequence = remove_urls(sequence)
                sequence = remove_html_tags(sequence)
                sequence = fix_slangs(sequence)
                sequence = decontracted(sequence)
                sequence = sequence.lower()
                encoded_input = tokenizer.encode(sequence)
                length = len(encoded_input)
                self.data_pool[id][0:length] = torch.tensor(encoded_input)
                self.mask_pool[id][0:length] = 1
                if self.is_training:
                    self.label_pool[id] = int(row["target"])
                    if np.random.randint(0, 10) < 9:
                        self.training_indices.append(id)
                    else:
                        self.dev_indices.append(id)
                else:
                    self.training_indices.append(id)

    def train(self):
        self.is_dev = False

    def eval(self):
        self.is_dev = True

    def __len__(self):
        if self.is_dev:
            return len(self.dev_indices)
        else:
            return len(self.training_indices)

    def __getitem__(self, item):
        if self.is_training:
            if self.is_dev:
                return self.data_pool[self.dev_indices[item]], self.mask_pool[self.dev_indices[item]], self.label_pool[self.dev_indices[item]]
            else:
                return self.data_pool[self.training_indices[item]], self.mask_pool[self.training_indices[item]], self.label_pool[self.training_indices[item]]
        else:
            return self.training_indices[item], self.data_pool[self.training_indices[item]], self.mask_pool[self.training_indices[item]]


class NaiveLSTMBaselineClassifier(nn.Module):
    def __init__(self):
        super(NaiveLSTMBaselineClassifier, self).__init__()
        self.word_embeddings = nn.Embedding(60000, 256)
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, encoder, input_ids, attention_mask):
        encoded, _ = encoder(input=self.word_embeddings(input_ids))
        len_selector = attention_mask.sum(dim=-1)
        encoded = (encoded * attention_mask.to(torch.float).unsqueeze(dim=-1)).sum(dim=1) / len_selector.unsqueeze(dim=-1)

        log_prediction = self.classifier_head(encoded)
        return log_prediction


class RoBERTaClassifierHead(nn.Module):
    def __init__(self, config: RobertaConfig):
        super(RoBERTaClassifierHead, self).__init__()
        self.classifier_head = nn.Sequential(
            nn.Linear(config.hidden_size, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, encoder: RobertaModel, input_ids, attention_mask):
        encoded = encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        log_prediction = self.classifier_head(encoded)
        return log_prediction


def main():
    if not os.path.exists("./checkpoints"):
        os.mkdir("checkpoints")

    parser = argparse.ArgumentParser()
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser.add_argument(
        "--eval_mode",
        default=False,
        type=str2bool,
        required=False,
        help="Test or train the model",
    )
    parser.add_argument(
        "--baseline",
        default=False,
        type=str2bool,
        required=False,
        help="use the baseline or the transformers model",
    )
    parser.add_argument(
        "--load_weights",
        default=True,
        type=str2bool,
        required=False,
        help="Load the pretrained weights or randomly initialize the model",
    )
    parser.add_argument(
        "--iter_per",
        default=4,
        type=int,
        required=False,
        help="cumulative gradient iteration cycle",
    )

    args = parser.parse_args()
    directory_identifier = args.__str__().replace(" ", "") \
        .replace("iter_per=1,", "") \
        .replace("iter_per=2,", "") \
        .replace("iter_per=4,", "") \
        .replace("iter_per=8,", "") \
        .replace("iter_per=16,", "") \
        .replace("iter_per=32,", "") \
        .replace("iter_per=64,", "") \
        .replace("iter_per=128,", "") \
        .replace("eval_mode=True", "eval_mode=False")


    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    suffix = "roberta"
    if args.baseline:
        suffix = "naive"
        tokenizer = NaiveTokenizer()
    try:
        dataset, test_dataset, tokenizer = torch.load(open("checkpoints/dataset-%s.pyc" % suffix, "rb"))
        dev_dataset, _, _ = torch.load(open("checkpoints/dataset-%s.pyc" % suffix, "rb"))
    except:
        dataset = DisasterTweetsClassificationDataset(tokenizer, "data/train.csv", "train")
        test_dataset = DisasterTweetsClassificationDataset(tokenizer, "data/test.csv", "test")
        torch.save((dataset, test_dataset, tokenizer), open("checkpoints/dataset-%s.pyc" % suffix, "wb"))
        dev_dataset, _, _  = torch.load(open("checkpoints/dataset-%s.pyc" % suffix, "rb"))
    dev_dataset.eval()

    if args.baseline:
        encoder = nn.LSTM(256, 256, 1, batch_first=True, )
        model = NaiveLSTMBaselineClassifier()
    else:
        if args.load_weights:
            encoder = RobertaModel.from_pretrained("roberta-base")
            model = RoBERTaClassifierHead(encoder.config)
        else:
            config = RobertaConfig.from_pretrained("roberta-base")
            encoder = RobertaModel(config=config)
            model = RoBERTaClassifierHead(config)
    encoder.cuda()
    model.cuda()
    dataloader = datautils.DataLoader(
        dataset, batch_size=64 // args.iter_per, shuffle=True,
        num_workers=16, drop_last=False, pin_memory=True
    )
    dev_dataloader = datautils.DataLoader(
        dev_dataset, batch_size=64 // args.iter_per, shuffle=True,
        num_workers=16, drop_last=False, pin_memory=True
    )
    test_dataloader = datautils.DataLoader(
        test_dataset, batch_size=64 // args.iter_per, shuffle=False,
        num_workers=16, drop_last=False, pin_memory=True
    )

    if args.eval_mode:
        correct = 0
        all = 0
        encoder_, model_ = torch.load("checkpoints/%s" % directory_identifier)
        encoder.load_state_dict(encoder_)
        model.load_state_dict(model_)
        encoder.eval()
        with torch.no_grad():
            for ids, mask, label in dev_dataloader:
                ids, mask, label = ids.cuda(), mask.cuda(), label.cuda()
                prediction = model(encoder, ids, mask).argmax(dim=-1)
                correct += (prediction == label).to(torch.long).sum().item()
                all += mask.shape[0]
        print("dev acc:", correct / all)
        opt = AdamW(lr=1e-6, weight_decay=0.05, params=list(encoder.parameters()) + list(model.parameters()))
        encoder.train()
        iter_num = 0
        LOSS = []
        for _ in range(5):
            iterator = tqdm.tqdm(dev_dataloader)
            for ids, mask, label in iterator:
                ids, mask, label = ids.cuda(), mask.cuda(), label.cuda()
                log_prediction = model(encoder, ids, mask)
                loss = -log_prediction[torch.arange(ids.size(0)).cuda(), label].mean()
                if iter_num % args.iter_per == 0:
                    opt.zero_grad()
                (loss / args.iter_per).backward()
                LOSS.append(loss.item())
                if len(LOSS) > 10:
                    iterator.write("loss=%f" % np.mean(LOSS))
                    LOSS = []
                if iter_num % args.iter_per == args.iter_per - 1:
                    opt.step()
                iter_num += 1
        encoder.eval()
        with torch.no_grad():
            for ids, mask, label in dev_dataloader:
                ids, mask, label = ids.cuda(), mask.cuda(), label.cuda()
                prediction = model(encoder, ids, mask).argmax(dim=-1)
                correct += (prediction == label).to(torch.long).sum().item()
                all += mask.shape[0]
        print("dev acc rectified:", correct / all)

        with torch.no_grad():
            with open("submission.csv", "w") as fout:
                print("id,target", file=fout)
                for id, ids, mask in test_dataloader:
                    ids, mask = ids.cuda(), mask.cuda()
                    prediction = model(encoder, ids, mask).argmax(dim=-1)
                    for i in range(id.size(0)):
                        print("%d,%d" % (id[i], prediction[i]), file=fout)
        exit()
    if args.baseline:
        lr = 5e-4
    elif args.load_weights:
        lr = 1e-6
    else:
        lr = 5e-6
    opt = AdamW(lr=lr, weight_decay=0.10 if args.baseline else 0.05, params=list(encoder.parameters())+list(model.parameters()))
    flog = open("checkpoints/log-%s.txt" % directory_identifier, "w")
    flog.close()
    flogeval = open("checkpoints/evallog-%s.txt" % directory_identifier, "w")
    flogeval.close()
    iter_num = 0
    for epoch_idx in range(5 if args.baseline else 10):
        flog = open("checkpoints/log-%s.txt" % directory_identifier, "a")
        flogeval = open("checkpoints/evallog-%s.txt" % directory_identifier, "a")
        LOSS = []
        encoder.train()
        iterator = tqdm.tqdm(dataloader)
        for ids, mask, label in iterator:
            ids, mask, label = ids.cuda(), mask.cuda(), label.cuda()
            log_prediction = model(encoder, ids, mask)
            loss = -log_prediction[torch.arange(ids.size(0)).cuda(), label].mean()
            if iter_num % args.iter_per == 0:
                opt.zero_grad()
            (loss / args.iter_per).backward()
            LOSS.append(loss.item())
            if len(LOSS) > 10:
                iterator.write("loss=%f" % np.mean(LOSS))
                print("%f" % np.mean(LOSS), file=flog)
                LOSS = []
            if iter_num % args.iter_per == args.iter_per - 1:
                opt.step()
            iter_num += 1
        EVALLOSS = []
        encoder.eval()
        iterator = tqdm.tqdm(dev_dataloader)
        with torch.no_grad():
            for ids, mask, label in iterator:
                ids, mask, label = ids.cuda(), mask.cuda(), label.cuda()
                log_prediction = model(encoder, ids, mask)
                loss = -log_prediction[torch.arange(ids.size(0)).cuda(), label].mean()
                EVALLOSS.append(loss.item())
        iterator.write("evalloss-%d=%f" % (epoch_idx, np.mean(EVALLOSS)))
        print("%f" % np.mean(EVALLOSS), file=flogeval)

        flog.close()
        flogeval.close()
        torch.save((encoder.state_dict(), model.state_dict()), "checkpoints/%s" % directory_identifier)
if __name__ == "__main__":
    main()