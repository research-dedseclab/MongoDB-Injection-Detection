import csv
import re
import os

class FeatureExtractor:
    def __init__(self, query, payload_file="payload.txt"):
        self.query = query
        self.whitespace_chars = (
            "\u0009\u000A\u000B\u000C\u000D\u0020\u0085\u00A0\u1680\u180E"
            "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
            "\u200A\u2028\u2029\u202F\u205F\u3000"
        )
        whitespace_pattern = "[" + self.whitespace_chars + "]+"
        self.tempQuery = re.sub(whitespace_pattern, "", query)
        self.payload_file = payload_file

    def isNullComparison(self):
        return int("null" in self.query or "!= null" in self.query)

    def containsEmptyString(self):
        return int('""' in self.query or "''" in self.query or "' '" in self.query or '" "' in self.query)

    def containsRegexTrue(self):
        return int(any(p in self.tempQuery for p in ["/.*/", "/./", "/.", ".*", "^", "/@/", '""']))

    def containsPayload(self):
        if not os.path.isfile(self.payload_file):
            return 0
        with open(self.payload_file, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = re.sub("[" + self.whitespace_chars + "]", "", line.strip())
                if cleaned and cleaned in self.tempQuery:
                    return 1
        return 0

    def containsRegex(self):
        return int("$regex" in self.query)
    
    def doesAlterCollection(self):
        return int("dropDatabase" in self.query or "drop" in self.query or "remove" in self.query)

    def elementQueryOperations(self):
        return int("$exists" in self.tempQuery or "$type" in self.tempQuery)

    def containsExtremes(self):
        ops = ["zzz", "~", "foo", "bar", "invalid", "dummy username", "dummy password", "ObjectId(9999999999999999999999999)", 'ObjectId("000000000000000000000000")', "ObjectId(0)", "ObjectId(1)", "ObjectId(000)", "123", "000"]
        return int(any(op in self.tempQuery for op in ops))

    def containsCondition(self):
        ops = ["$eq", "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin", "$not"]
        return int(any(op in self.tempQuery for op in ops))

    def containsLogicalOp(self):
        return int(any(op in self.query for op in ["$or", "$and", "$not", "$nor"]))
    
    def containsAlwaysTrue(self):
        return int(any(op in self.query for op in ["'1'=='1'", "1==1", "''==''", "'true'=='true'", "'hacked'=='hacked'", "|| true", "|| 1==1", "|| 'a'=='a", "> ''", ">= ''", '> ""', '>= ""', "> '0'", "== '0'"]))
    
    def containsAdminOps(self):
        return int(any(op in self.query for op in ["serverStatus", "killOp", "setProfilingLevel", "runCommand", "shutDownServer", "stats", "logout", "adminCommand", "hostInfo", "isMaster", "shutdown", "fsuncLock()"]))

    def containsJS(self):
        return int("$where" in self.query)
    
    def containsSleep(self):
        return int("sleep(" in self.query)

    def containsReturn(self):
        return int(any(p in self.query for p in [";return", "return 1", "return true", "return(true)", "return 0"]))

    def containsEmptyArray(self):
        return int("{}" in self.query or "[]" in self.query)

    def containsComments(self):
        return int("//" in self.query)

    def isNewQuery(self):
        return int(";db." in self.query or ";" in self.query)

    def isWhileTrue(self):
        return int("while(true)" in self.query or "while(1)" in self.query or "while(!0)" in self.query)

    def extract_all_features(self):
        return [
            self.isNullComparison(),
            self.containsEmptyString(),
            self.containsRegexTrue(),
            self.containsPayload(),
            self.containsRegex(),
            self.doesAlterCollection(),
            self.elementQueryOperations(),
            self.containsExtremes(),
            self.containsCondition(),
            self.containsLogicalOp(),
            self.containsAlwaysTrue(),
            self.containsAdminOps(),
            self.containsJS(),
            self.containsSleep(),
            self.containsReturn(),
            self.containsEmptyArray(),
            self.containsComments(),
            self.isNewQuery(),
            self.isWhileTrue()
        ]

feature_names = [
    "isNullComparison", "containsEmptyString", "containsRegexTrue", "containsPayload",
    "containsRegex", "doesAlterCollection", "elementQueryOperations", "containsExtremes", "containsCondition",
    "containsLogicalOp", "containsAlwaysTrue", "containsAdminOps", "containsJS", "containsSleep", "containsReturn", 
    "containsEmptyArray", "containsComments", "isNewQuery", "isWhileTrue"
]

# Load CSV and extract features
input_csv = 'our_dataset_train_queries.csv'   
output_csv = 'our_dataset_train_queries_features.csv'

with open(input_csv, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    writer.writerow(feature_names + ['label'])  # Header row

    for row in reader:
        query = row['query']
        label = row.get('label', '')
        extractor = FeatureExtractor(query)
        features = extractor.extract_all_features()
        writer.writerow(features + [label])
