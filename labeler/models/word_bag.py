"""Interface to a sqlite database that stores word frequencies for each
label as defined in the config.ini file"""

import sqlite3
import re
import collections
import itertools
import labeler.models.tag_manager as tm
import labeler.models.config as cfg

DB_PATH = 'labeler/data/word_bag.db'

SELECT_TABLE = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name=?;
"""
CREATE_LABELS_TABLE = """
    CREATE TABLE IF NOT EXISTS label (
        id      INTEGER     PRIMARY KEY,
        name    TEXT        UNIQUE
    );
"""
CREATE_FREQUENCY_TABLE = """
    CREATE TABLE IF NOT EXISTS frequency (
        id      INTEGER     PRIMARY KEY,
        word    TEXT,
        label   INTEGER,
        count   INTEGER,
        UNIQUE(word, label)
    );
"""
CREATE_BIGRAM_TABLE = """
    CREATE TABLE IF NOT EXISTS bigram (
        id      INTEGER     PRIMARY KEY,
        label1  INTEGER,
        label2  INTEGER,
        count   INTEGER,
        UNIQUE(label1, label2)
    );
"""
INSERT_LABEL = """
    INSERT OR IGNORE INTO label (name) VALUES (?);
"""
INSERT_BIGRAMS = """
    INSERT OR IGNORE INTO bigram (label1, label2, count) VALUES (?, ?, ?);
"""
SELECT_LABELS = """
    SELECT name, id
    FROM label;
"""
UPDATE_FREQUENCY = """
    INSERT OR REPLACE INTO frequency (
        word, label, count
    ) VALUES (
        :word, :label, COALESCE(
            (SELECT count + :count
             FROM frequency
             WHERE word=:word AND label=:label
            ), :count
        )
    );
"""
UPDATE_BIGRAMS = """
    UPDATE OR IGNORE bigram
    SET count=:count
    WHERE label1=:label1 AND label2=:label2;
"""
GET_PROBABILITIES = """
SELECT
    counts.id AS id,
    (counts.label_count + 1.0) / (counts.total_count + (SELECT COUNT(*) FROM label)) AS probability
FROM (
        SELECT
            label.id AS id,
            COALESCE(word_counts.count, 0) AS label_count,
            total_counts.total AS total_count
        FROM label
        LEFT JOIN (
                SELECT f.label, f.count AS count FROM frequency f WHERE f.word=:word
            ) word_counts ON label.id=word_counts.label
        LEFT JOIN (
                SELECT SUM(f.count) AS total FROM frequency f WHERE f.word=:word
            ) total_counts
    ) counts;
"""


class WordBag:

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.conn.close()

    def __init__(self, initialize=False):
        self.conn = sqlite3.connect(DB_PATH)
        self.c = self.conn.cursor()

        # Create required tables if they don't exists
        # Label table
        self.c.execute(CREATE_LABELS_TABLE)
        self.c.executemany(
            INSERT_LABEL, [(label,) for label in cfg.Config.labels.keys()]
        )
        # Frequency table
        self.c.execute(CREATE_FREQUENCY_TABLE)
        # Bigram table
        self.c.execute(CREATE_BIGRAM_TABLE)
        lbl_cnt = len(cfg.Config.labels)
        for label1, label2 in itertools.product(range(lbl_cnt), range(lbl_cnt)):
            self.c.execute(INSERT_BIGRAMS, (label1, label2, 0))

    def get_labels(self):
        self.c.execute(SELECT_LABELS).fetchall()

    def label_frequency(self, label_dict):
        """Give the result of reading a labeled JSON file"""
        label_id = dict(self.c.execute(SELECT_LABELS).fetchall())

        count_dict = collections.defaultdict(int)
        for row, column_dict in label_dict[tm.CONTENT].items():
            for column, word_list in column_dict.items():
                for word in word_list:
                    try:
                        clean_word = self.clean_digits(word['word'])
                        str_tag = word['tags'][0]
                        int_tag = label_id[str_tag]
                        count_dict[(clean_word, int_tag)] += 1
                    except (TypeError, IndexError):
                        continue

        for (clean_word, int_tag), count in count_dict.items():
            self.c.execute(
                UPDATE_FREQUENCY, {
                    'word': clean_word,
                    'label': int_tag,
                    'count': count
                }
            )
        self.conn.commit()

    def clean_digits(self, string):
        """Replaces all digits in the string with zeros for standardization"""
        return re.sub('\d', '0', string)

    def label_probabilities(self, word):
        results = self.c.execute(GET_PROBABILITIES, {'word': word}).fetchall()
        print([x[1] for x in sorted(results)])
