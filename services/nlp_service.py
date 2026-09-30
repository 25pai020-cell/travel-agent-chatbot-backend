import re
from typing import Dict, Any, List, Optional

# Load spaCy NLP model (offline) with fallback
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
        except Exception:
            nlp = None
except Exception:
    nlp = None

class SimpleToken:
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower()

class SimpleDoc:
    def __init__(self, text: str):
        self.text = text
        self.tokens = [SimpleToken(w) for w in re.findall(r'\b\w+\b', text)]
        self.ents = []
        # Basic regex-based money/date entity extraction fallback (requires explicit currency symbol or multiplier)
        money_matches = re.finditer(
            r'(?:₹|rs\.?|inr|rupees)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?|\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)\b',
            text,
            re.IGNORECASE
        )
        for m in money_matches:
            val_text = m.group(1) or m.group(3)
            if val_text:
                class Ent:
                    def __init__(self, t, l):
                        self.text = t
                        self.label_ = l
                self.ents.append(Ent(m.group(0), "MONEY"))

    def __iter__(self):
        return iter(self.tokens)

# Word to number mapping for spoken numbers
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15
}

# Domain Interest Ontologies mapped to dataset trip_types & characteristics
INTEREST_KEYWORDS: Dict[str, List[str]] = {
    "nature": ["nature", "scenic", "landscape", "greenery", "mountain", "valley", "lake", "waterfall", "flora", "natural", "forest"],
    "adventure": ["adventure", "trekking", "trek", "hiking", "hike", "rafting", "camping", "safari", "thrill", "climbing", "sports", "zipline", "paragliding"],
    "beach": ["beach", "beaches", "coastal", "coast", "sea", "ocean", "island", "sand", "seashore", "watersports"],
    "cultural": ["cultural", "culture", "heritage", "historical", "history", "monument", "monuments", "palace", "fort", "museum", "archaeological", "tradition"],
    "wildlife": ["wildlife", "safari", "national park", "sanctuary", "animal", "tiger", "birds", "bird watching", "fauna"],
    "hills": ["hills", "hill station", "hillstation", "snow", "snowfall", "peaks", "cooler"],
    "religious": ["religious", "spiritual", "pilgrimage", "temple", "temples", "shrine", "church", "mosque", "monastery", "holy", "sacred", "darshan"],
    "food": ["food", "culinary", "cuisine", "street food", "dining", "delicacy", "gourmet", "tasting", "cafe"],
    "wellness": ["wellness", "spa", "ayurveda", "yoga", "meditation", "retreat", "rejuvenation", "relaxing", "peace", "calm"],
    "nightlife": ["nightlife", "party", "parties", "clubbing", "pubs", "bars", "music festival", "evening life"],
    "photography": ["photography", "photo", "photogenic", "picturesque", "viewpoint", "viewpoints", "scenery", "camera"],
    "romantic": ["romantic", "honeymoon", "candlelight", "sunset", "getaway for couple"]
}

# Seasons & Month Mapping
SEASON_KEYWORDS: Dict[str, List[str]] = {
    "winter": ["winter", "december", "january", "february", "cold", "chilly"],
    "summer": ["summer", "march", "april", "may", "june", "hot"],
    "monsoon": ["monsoon", "rainy", "july", "august", "september", "rains"],
    "post-monsoon": ["post-monsoon", "autumn", "fall", "october", "november", "spring"]
}

# Group Types
GROUP_KEYWORDS: Dict[str, List[str]] = {
    "solo": ["solo", "alone", "myself", "single traveler", "by myself"],
    "couple": ["couple", "honeymoon", "partner", "wife", "husband", "girlfriend", "boyfriend", "romantic"],
    "friends": ["friends", "buddies", "gang", "group of friends", "college friends", "colleagues"],
    "family": ["family", "parents", "kids", "children", "elderly", "relatives"]
}


class NLPPreferenceExtractor:
    """
    Offline NLP Preference Extraction Engine
    Combines spaCy linguistic parsing, Regex pattern matching, and domain-specific rule dictionaries.
    """

    def __init__(self):
        self.nlp = nlp

    def extract_days(self, text: str, doc) -> Optional[int]:
        """Extract trip duration in days."""
        lower_text = text.lower()

        # Check special duration words first
        if "weekend" in lower_text:
            return 2
        if "a week" in lower_text or "one week" in lower_text:
            return 7
        if "fortnight" in lower_text or "two weeks" in lower_text:
            return 14

        # Regex: e.g. "4-day", "4 days", "4 day", "4days", "4 d"
        day_match = re.search(r'\b(\d+)\s*[- ]*(?:day|days|d)\b', lower_text)
        if day_match:
            days = int(day_match.group(1))
            if 1 <= days <= 60:
                return days

        # Regex: e.g. "four-day", "four days", "for four days"
        words_pattern = r'\b(' + '|'.join(WORD_TO_NUM.keys()) + r')\s*[- ]*(?:day|days)\b'
        word_match = re.search(words_pattern, lower_text)
        if word_match:
            word = word_match.group(1)
            return WORD_TO_NUM.get(word)

        # Regex: e.g. "for 4 days", "spend 5 days"
        for_days = re.search(r'\b(?:for|spend|plan)\s+(\d+)\s*(?:days?)?\b', lower_text)
        if for_days:
            days = int(for_days.group(1))
            if 1 <= days <= 60:
                return days

        # spaCy entity analysis for DATE / CARDINAL
        for ent in doc.ents:
            if ent.label_ in ("DATE", "CARDINAL"):
                ent_text = ent.text.lower()
                m = re.search(r'(\d+)', ent_text)
                if m and any(unit in ent_text for unit in ["day", "night", "week"]):
                    val = int(m.group(1))
                    if "week" in ent_text:
                        val *= 7
                    return val

        return None

    def extract_budget(self, text: str, doc) -> Optional[float]:
        """Extract total budget in INR."""
        lower_text = text.lower().replace(",", "")
        days_in_text = self.extract_days(text, doc)

        # 1. Look for patterns with rupee symbol ₹ or rs / inr / rupees / rupiess / rupes.
        # Symbol BEFORE number: e.g., "₹20000", "rs 20000", "inr 20000"
        curr_before_match = re.search(r'(?:₹|rs\.?|inr|rupees?|rupiess?|rupes)\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?', lower_text)
        if curr_before_match:
            num = float(curr_before_match.group(1))
            multiplier = curr_before_match.group(2)
            if multiplier == "k" or multiplier == "thousand":
                num *= 1000
            elif multiplier in ("lakh", "lakhs"):
                num *= 100000
            return num

        # Symbol AFTER number: e.g., "10000 rupees", "10000 rupiess", "10000 rs", "10000 inr"
        curr_after_match = re.search(r'(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?\s*(?:₹|rs\.?|inr|rupees?|rupiess?|rupes)\b', lower_text)
        if curr_after_match:
            num = float(curr_after_match.group(1))
            multiplier = curr_after_match.group(2)
            if multiplier == "k" or multiplier == "thousand":
                num *= 1000
            elif multiplier in ("lakh", "lakhs"):
                num *= 100000
            return num

        # 2. Look for "under 20k", "under 20,000", "budget 20k", "around 15000", "max 25000", "upto 30000"
        budget_prep_match = re.search(r'\b(?:under|below|within|upto|up\s*to|max|around|budget\s*(?:of)?|budget:?)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?\b', lower_text)
        if budget_prep_match:
            num = float(budget_prep_match.group(1))
            multiplier = budget_prep_match.group(2)
            if multiplier == "k" or multiplier == "thousand":
                num *= 1000
            elif multiplier in ("lakh", "lakhs"):
                num *= 100000
            if days_in_text and num == float(days_in_text) and not multiplier:
                pass
            else:
                return num

        # 3. Handle shorthand "20k" or "20 thousand" directly
        k_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:k|thousand)\b', lower_text)
        if k_match:
            return float(k_match.group(1)) * 1000

        # 4. Handle "1 lakh", "2 lakhs"
        lakh_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)\b', lower_text)
        if lakh_match:
            return float(lakh_match.group(1)) * 100000

        # 5. spaCy MONEY entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                ent_lower = ent.text.lower()
                if any(kw in ent_lower for kw in ["₹", "rs", "inr", "rupee", "k", "thousand", "lakh"]):
                    m = re.search(r'(\d+(?:\.\d+)?)', ent.text.replace(",", ""))
                    if m:
                        val = float(m.group(1))
                        if "k" in ent_lower:
                            val *= 1000
                        elif "lakh" in ent_lower:
                            val *= 100000
                        if days_in_text and val == float(days_in_text):
                            continue
                        return val

        # 6. Fallback: Standalone numbers >= 500 in trip context distinct from days
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', lower_text)
        for num_str in numbers:
            val = float(num_str)
            if days_in_text and val == float(days_in_text):
                continue
            if val >= 500:
                return val

        return None

    def extract_interests(self, text: str, doc) -> List[str]:
        """Extract travel interests / themes from text using token lemmatization and rule matching."""
        lower_text = text.lower()
        lemmas = [token.lemma_.lower() for token in doc]
        detected = set()

        for category, keywords in INTEREST_KEYWORDS.items():
            for kw in keywords:
                # Check direct phrase in text or lemmatized match
                if kw in lower_text or kw in lemmas:
                    detected.add(category)
                    break

        return sorted(list(detected))

    def extract_season(self, text: str) -> Optional[str]:
        """Extract travel season or implied season from months."""
        lower_text = text.lower()
        for season, keywords in SEASON_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
                    return season
        return None

    def extract_group_type(self, text: str) -> Optional[str]:
        """Extract traveling group or companion type."""
        lower_text = text.lower()
        for group, keywords in GROUP_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
                    return group
        return None

    def extract_destination(self, text: str) -> Optional[str]:
        """
        Extract explicit destination mentioned or chosen by user.
        Supports fuzzy matching and comprehensive alias dictionary for all Indian destinations.
        """
        import difflib

        lower_text = text.lower().strip()

        months_and_stopwords = {
            "january", "february", "march", "april", "may", "june", "july", "august", 
            "september", "october", "november", "december", "winter", "summer", "monsoon", 
            "autumn", "spring", "weekend", "friends", "family", "budget", "india", "trip", 
            "tour", "vacation", "holiday", "holidaying", "morning", "afternoon", "evening",
            "remove", "add", "temple", "temples", "adventure", "nature", "beach", "cultural"
        }

        # Comprehensive Indian Destination Alias & Misspelling Dictionary
        INDIAN_DESTINATION_ALIASES = {
            "manali": "Manali", "manale": "Manali", "manali trip": "Manali", "old manali": "Manali",
            "goa": "Goa", "goaa": "Goa", "north goa": "Goa", "south goa": "Goa",
            "jaipur": "Jaipur", "jaipure": "Jaipur", "pink city": "Jaipur",
            "udaipur": "Udaipur", "udaipure": "Udaipur", "lake city": "Udaipur",
            "munnar": "Munnar", "munar": "Munnar",
            "kerala": "Kerala", "kerala backwaters": "Kerala", "alleppey": "Alleppey", "alappuzha": "Alleppey",
            "shimla": "Shimla", "shila": "Shimla", "shimlaa": "Shimla",
            "ladakh": "Ladakh", "leh": "Ladakh", "leh ladakh": "Ladakh",
            "agra": "Agra", "taj mahal": "Agra",
            "rishikesh": "Rishikesh", "haridwar": "Haridwar",
            "varanasi": "Varanasi", "kashi": "Varanasi", "banaras": "Varanasi",
            "coorg": "Coorg", "kodagu": "Coorg",
            "ooty": "Ooty", "udhagamandalam": "Ooty",
            "kodaikanal": "Kodaikanal", "kodai": "Kodaikanal",
            "hampi": "Hampi", "gokarna": "Gokarna", "varkala": "Varkala",
            "darjeeling": "Darjeeling", "gangtok": "Gangtok", "sikkim": "Gangtok",
            "shillong": "Shillong", "cherrapunji": "Cherrapunji", "meghalaya": "Shillong",
            "pondicherry": "Pondicherry", "puducherry": "Pondicherry",
            "amritsar": "Amritsar", "golden temple": "Amritsar",
            "wayanad": "Wayanad", "mahabaleshwar": "Mahabaleshwar",
            "jaisalmer": "Jaisalmer", "gulmarg": "Gulmarg", "auli": "Auli",
            "spiti": "Spiti Valley", "spiti valley": "Spiti Valley",
            "nainital": "Nainital", "mussoorie": "Mussoorie", "kasol": "Kasol",
            "delhi": "Delhi", "mumbai": "Mumbai", "bangalore": "Bangalore", "bengaluru": "Bangalore",
            "kolkata": "Kolkata", "hyderabad": "Hyderabad", "chennai": "Chennai", "pune": "Pune",
            "kochi": "Kochi", "cochin": "Kochi", "chikmagalur": "Chikmagalur",
            "tawang": "Tawang", "khajjiar": "Khajjiar", "kanyakumari": "Kanyakumari",
            "dharamshala": "Dharamshala", "mcleodganj": "Dharamshala",
            "mathura": "Mathura", "vrindavan": "Vrindavan", "rameswaram": "Rameswaram",
            "bodhgaya": "Bodh Gaya", "bodh gaya": "Bodh Gaya",
            "kaziranga": "Kaziranga", "ranthambore": "Ranthambore", "kanha": "Kanha",
            "khajuraho": "Khajuraho", "orchha": "Orchha", "rann of kutch": "Rann of Kutch", "kutch": "Rann of Kutch"
        }

        # 1. Check explicit selection prefixes (e.g. "Selected Destination = Manali", "I choose Jaipur", "Plan for Goa")
        patterns = [
            r'(?:selected destination\s*[:=]\s*)([a-zA-Z\s]+)',
            r'(?:user\s*[:=]\s*)([a-zA-Z\s]+)',
            r'(?:i choose|i choose to go to|let\'s go to|let us go to|plan for|trip to|explore|visit)\s+([a-zA-Z\s]+?)(?:\s*[\.\!\,\(]|$)',
            r'\b(?:in|to)\s+([A-Z][a-zA-Z]+)\b'
        ]

        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip().lower()
                cand_clean = re.sub(r'\s*\([^)]*\)', '', cand).strip()
                if len(cand_clean) >= 3 and cand_clean not in months_and_stopwords:
                    if cand_clean in INDIAN_DESTINATION_ALIASES:
                        return INDIAN_DESTINATION_ALIASES[cand_clean]
                    # Fuzzy match candidate against alias keys
                    matches = difflib.get_close_matches(cand_clean, list(INDIAN_DESTINATION_ALIASES.keys()), n=1, cutoff=0.7)
                    if matches:
                        return INDIAN_DESTINATION_ALIASES[matches[0]]
                    return cand_clean.title()

        # 2. Direct exact search in text against alias keys (sorted by key length descending)
        for alias in sorted(INDIAN_DESTINATION_ALIASES.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', lower_text):
                return INDIAN_DESTINATION_ALIASES[alias]

        # 3. Token-level fuzzy matching for short prompts or misspelled queries (e.g. "manale", "jaipure", "munar", "goaa", "shila")
        tokens = [w.strip() for w in re.split(r'[\s,\.!]+', lower_text) if w.strip()]
        for tok in tokens:
            if len(tok) >= 3 and tok not in months_and_stopwords:
                matches = difflib.get_close_matches(tok, list(INDIAN_DESTINATION_ALIASES.keys()), n=1, cutoff=0.75)
                if matches:
                    return INDIAN_DESTINATION_ALIASES[matches[0]]

        return None


    def extract_modifications(self, text: str) -> Dict[str, Any]:
        """
        1️⃣4️⃣ User Modifies the Itinerary 🔄
        Detects REMOVE (e.g. temples, museums) and ADD (e.g. adventure, nature) intents.
        Example: "Remove temples and add more adventure."
        -> remove_categories: ["temple", "religious"], add_interests: ["adventure"]
        """
        lower = text.lower()
        remove_categories = set()
        add_interests = set()

        # REMOVE intent detection
        remove_patterns = [
            r'(?:remove|delete|exclude|no|without|drop|don\'t include|dont include)\s+([a-zA-Z\s,]+)',
            r'(?:replace)\s+([a-zA-Z\s]+)\s+with',
        ]
        for pat in remove_patterns:
            m = re.search(pat, lower)
            if m:
                target = m.group(1).strip()
                if "temple" in target or "temples" in target or "religious" in target or "mandir" in target:
                    remove_categories.update(["temple", "religious", "spiritual"])
                if "museum" in target or "museums" in target:
                    remove_categories.update(["museum", "cultural"])
                if "market" in target or "shopping" in target:
                    remove_categories.update(["shopping", "market"])

        # ADD intent detection
        add_patterns = [
            r'(?:add|include|more|want|with|replace\s+[a-zA-Z\s]+\s+with)\s+([a-zA-Z\s,]+)'
        ]
        for pat in add_patterns:
            m = re.search(pat, lower)
            if m:
                target = m.group(1).strip()
                for cat, kws in INTEREST_KEYWORDS.items():
                    if cat in target or any(kw in target for kw in kws):
                        add_interests.add(cat)

        return {
            "is_modification": bool(remove_categories or add_interests),
            "remove_categories": sorted(list(remove_categories)),
            "add_interests": sorted(list(add_interests))
        }

    def extract_preferences(self, text: str) -> Dict[str, Any]:
        """
        Main extraction pipeline.
        Processes user raw text and returns structured dictionary.
        """
        doc = self.nlp(text) if self.nlp else SimpleDoc(text)

        days = self.extract_days(text, doc)
        budget = self.extract_budget(text, doc)
        interests = self.extract_interests(text, doc)
        season = self.extract_season(text)
        group_type = self.extract_group_type(text)
        destination = self.extract_destination(text)
        modifications = self.extract_modifications(text)

        return {
            "days": days,
            "budget": budget,
            "interests": interests,
            "season": season,
            "group_type": group_type,
            "destination": destination,
            "is_modification": modifications["is_modification"],
            "remove_categories": modifications["remove_categories"],
            "add_interests": modifications["add_interests"]
        }


# Singleton instance for quick imports
nlp_extractor = NLPPreferenceExtractor()

