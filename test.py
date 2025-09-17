from collections import defaultdict
import re

def normalize(text: str) -> str:
    # Lowercase and remove non-alphanumeric for loose matching
    return re.sub(r'\W+', ' ', text.lower()).strip()

def diseases_with_all_symptoms(kg_triples, target_symptoms):
    target_set = {normalize(s) for s in target_symptoms}

    count_map = defaultdict(int)      # counts matches per disease
    matched_map = defaultdict(list)   # stores matching symptom strings

    for disease, rel, symptom in kg_triples:
        norm_sym = normalize(symptom)
        for t in target_set:
            # check if the target phrase is contained in the normalized symptom
            if t in norm_sym:
                count_map[disease] += 1
                matched_map[disease].append(symptom)
                break  # avoid double counting this triple for multiple targets

    # filter diseases where count equals number of target symptoms
    return {
        disease: matched_map[disease]
        for disease, c in count_map.items()
        if c == len(target_set)
    }

# Example usage
kg_triples = [
    ('Antibiotic-associated diarrhea', 'IS_SYMPTOM', 'fever'),
    ('Antibiotic-associated diarrhea', 'IS_SYMPTOM', 'cold'),
    ('Asthma', 'IS_SYMPTOM', 'developing a sinus headache or fever'),
    ('Endocarditis', 'IS_SYMPTOM', 'high fever'),
    ('Shigellosis', 'IS_SYMPTOM', 'brief fever')
]

target_symptoms = ["fever",'cold']
print(diseases_with_all_symptoms(kg_triples, target_symptoms))
