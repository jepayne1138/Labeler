"""Interface to a sqlite database that stores word frequencies for each
label as defined in the config.ini file"""

import os
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
CREATE_HEADER_TABLE = """
    CREATE TABLE IF NOT EXISTS header (
        id      INTEGER     PRIMARY KEY,
        name    TEXT        UNIQUE
    );
"""
CREATE_HEADER_FREQUENCY_TABLE = """
    CREATE TABLE IF NOT EXISTS header_frequency (
        id      INTEGER     PRIMARY KEY,
        header  INTEGER,
        label   INTEGER,
        count   INTEGER,
        UNIQUE(header, label)
    );
"""
CREATE_TRAINED_TABLE = """
    CREATE TABLE IF NOT EXISTS trained (
        id      INTEGER     PRIMARY KEY,
        name    TEXT        UNIQUE
    );
"""
INSERT_LABEL = """
    INSERT OR IGNORE INTO label (name) VALUES (?);
"""
INSERT_BIGRAMS = """
    INSERT OR IGNORE INTO bigram (label1, label2, count) VALUES (?, ?, ?);
"""
INSERT_FILENAME = """
    INSERT OR FAIL INTO trained (name) VALUES (?);
"""
INSERT_HEADER = """
    INSERT OR FAIL INTO header (name) VALUES (?);
"""
INSERT_HEADER_FREQUENCY = """
    INSERT OR FAIL INTO header_frequency (header, label, count) VALUES (?, ?, ?);
"""
SELECT_LABELS = """
    SELECT name, id
    FROM label;
"""
SELECT_HEADERS = """
    SELECT name, id
    FROM header;
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
UPDATE_HEADER_FREQUENCY = """
    UPDATE OR IGNORE header_frequency
    SET count=COALESCE(
        (SELECT count + :count
         FROM header_frequency
         WHERE header=:header AND label=:label
        ), :count
    )
    WHERE header=:header AND label=:label;
"""
UPDATE_BIGRAMS = """
    UPDATE OR IGNORE bigram
    SET count=:count
    WHERE label1=:label1 AND label2=:label2;
"""
GET_PROBABILITIES = """
    SELECT
        counts.id AS id,
        (counts.label_count + (1.0 / (SELECT COUNT(*) FROM label))) / (counts.total_count + 1.0) AS probability
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
                    SELECT COALESCE(SUM(f.count), 0) AS total FROM frequency f WHERE f.word=:word
                ) total_counts
        ) counts
    ORDER BY counts.id ASC;
"""
GET_BIGRAM_PROB_FIRST_COND_SECOND = """
    SELECT
        bigram.id AS id,
        bigram.label1 AS label1,
        bigram.label2 AS label2,
        (bigram.count + (1.0 / (SELECT COUNT(*) FROM label))) / (count.total_count + 1.0) AS probability
    FROM bigram
    LEFT JOIN (
            SELECT bigram.label2 AS label, sum(count) AS total_count FROM bigram GROUP BY label2
    ) count ON count.label=bigram.label2
    ORDER BY bigram.label1 ASC, bigram.label2 ASC;
"""
GET_BIGRAM_PROB_SECOND_COND_FIRST = """
    SELECT
        bigram.id AS id,
        bigram.label1 AS label1,
        bigram.label2 AS label2,
        (bigram.count + (1.0 / (SELECT COUNT(*) FROM label))) / (count.total_count + 1.0) AS probability
    FROM bigram
    LEFT JOIN (
            SELECT bigram.label1 AS label, sum(count) AS total_count FROM bigram GROUP BY label1
    ) count ON count.label=bigram.label1
    ORDER BY bigram.label2 ASC, bigram.label1 ASC;
"""


class WordBag:

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.conn.close()

    def __init__(self, initialize=False):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
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
            self.c.execute(INSERT_BIGRAMS, (label1 + 1, label2 + 1, 0))
        # Trained files table
        self.c.execute(CREATE_TRAINED_TABLE)
        # Header table
        self.c.execute(CREATE_HEADER_TABLE)
        self.c.executemany(
            INSERT_HEADER, [(header,) for header in cfg.Config.headers.keys()]
        )
        # Header frequency table
        self.c.execute(CREATE_HEADER_FREQUENCY_TABLE)
        header_cnt = len(cfg.Config.headers)
        for header, label in itertools.product(range(header_cnt), range(lbl_cnt)):
            self.c.execute(INSERT_HEADER_FREQUENCY, (header + 1, label + 1, 0))

    def get_labels(self):
        return self.c.execute(SELECT_LABELS).fetchall()

    def label_frequency(self, label_dict):
        """Give the result of reading a labeled JSON file"""
        label_id = dict(self.c.execute(SELECT_LABELS).fetchall())
        header_id = dict(self.c.execute(SELECT_HEADERS).fetchall())
        if tm.HEADER in label_dict:
            headers_map = {}
            for column, header_dict in label_dict[tm.HEADER].items():
                try:
                    headers_map[column] = header_id[header_dict['tags'][0]]
                except (IndexError, KeyError):
                    pass
        else:
            headers_map = None

        filename, _ = os.path.splitext(label_dict[tm.FILE])

        # Attempt to insert filename to record that the file has be trained
        try:
            self.c.execute(INSERT_FILENAME, (filename,))
        except sqlite3.IntegrityError:
            # Word bag already trained with this file
            return

        header_count_dict = collections.defaultdict(int)
        count_dict = collections.defaultdict(int)
        bigram_dict = collections.defaultdict(int)
        for row, column_dict in label_dict[tm.CONTENT].items():
            for column, word_list in column_dict.items():
                prev_word_int_tag = None
                for word in word_list:
                    try:
                        clean_word = self.clean_digits(word['word'])
                        str_tag = word['tags'][0]
                        int_tag = label_id[str_tag]
                        count_dict[(clean_word, int_tag)] += 1

                        # Add to header frequency table
                        if headers_map:
                            header_int = headers_map[column]
                            header_count_dict[(header_int, int_tag)] += 1

                        # Add to bigram table
                        if prev_word_int_tag is not None:
                            bigram_dict[(prev_word_int_tag, int_tag)] += 1

                        # Set current word to new prev_word
                        prev_word_int_tag = int_tag
                    except (TypeError, IndexError):
                        continue

        # Update monogram label counts
        for (clean_word, int_tag), count in count_dict.items():
            self.c.execute(
                UPDATE_FREQUENCY, {
                    'word': clean_word,
                    'label': int_tag,
                    'count': count
                }
            )

        # Update bigram label counts
        for (label1, label2), count in bigram_dict.items():
            self.c.execute(
                UPDATE_BIGRAMS, {
                    'label1': label1,
                    'label2': label2,
                    'count': count
                }
            )

        # Update the header counts
        for (header_int, int_tag), count in header_count_dict.items():
            self.c.execute(
                UPDATE_HEADER_FREQUENCY, {
                    'header': header_int,
                    'label': int_tag,
                    'count': count
                }
            )

        self.conn.commit()

    def clean_digits(self, string):
        """Replaces all digits in the string with zeros for standardization"""
        return re.sub('\d', '0', string)

    def raw_label_probabilities(self, word):
        results = self.c.execute(GET_PROBABILITIES, {'word': word}).fetchall()
        return results

    def bigram_prob_first_cond_second(self):
        results = self.c.execute(GET_BIGRAM_PROB_FIRST_COND_SECOND).fetchall()
        return results

    def bigram_prob_second_cond_first(self):
        results = self.c.execute(GET_BIGRAM_PROB_SECOND_COND_FIRST).fetchall()
        return results

    def label_probabilities(self, word):
        results = self.c.execute(GET_PROBABILITIES, {'word': word}).fetchall()
        print([x[1] for x in sorted(results)])
