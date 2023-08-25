import psycopg
import math
import time
import datetime
import logging
import sys
from multiprocessing import Process
import numpy as np
from numpy.linalg import norm
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    f'[%(threadName)s] [%(asctime)s] [%(levelname)s] --- %(message)s')
stdouthandler = logging.StreamHandler(sys.stdout)
stdouthandler.setLevel(logging.INFO)
stdouthandler.setFormatter(formatter)
logger.addHandler(stdouthandler)

THREAD_COUNT = 8


def get_doc_occ(issues, emails):
    logger.info(f"Starting doc_occ calculation")
    doc_dict = dict()

    for issue in issues:
        words = {word.strip('\''): len(occ.split(',')) for word, _, occ in [word.rpartition(':') for word in issue["description_summary_vector"].split(' ')]}
        for word in words:
            if word in doc_dict.keys():
                doc_dict[word] += 1
            else:
                doc_dict.update({word: 1})

    for email in emails:
        words = {word.strip('\''): len(occ.split(',')) for word, _, occ in [word.rpartition(':') for word in email["body_vector"].split(' ')]}
        for word in words:
            if word in doc_dict.keys():
                doc_dict[word] += 1
            else:
                doc_dict.update({word: 1})

    logger.info(f"Finished doc_occ calculation. Found {len(doc_dict)} words")
    return doc_dict


def cosine_similarity(first, second, doc_occ, total_docs):  # two tsvectors
    first_sep = {word.strip('\''): len(occ.split(',')) for word, _, occ in [word.rpartition(':') for word in first.split(' ')]}
    second_sep = {word.strip('\''): len(occ.split(',')) for word, _, occ in [word.rpartition(':') for word in second.split(' ')]}

    all_words = list(set(first_sep.keys()).union(set(second_sep.keys())))

    first_vec, second_vec = [], []
    for word in all_words:
        first_vec.append(math.log(total_docs / doc_occ[word]) * math.log(1 + first_sep.get(word, 0)))
        second_vec.append(math.log(total_docs / doc_occ[word]) * math.log(1 + second_sep.get(word, 0)))

    return np.dot(first_vec, second_vec)/(norm(first_vec)*norm(second_vec))


def loop(issues, doc_dict, emails, result_db_name):
    start = time.time()

    with psycopg.connect(
            host="localhost",
            dbname="relationsDB",
            user="postgres",
            password="UnsavePassword",
            port=5432) as conn:
        cur = conn.cursor(row_factory=psycopg.rows.dict_row)

        total_issues = len(issues)
        total_emails = len(emails)

        cnt = 0
        sims = []

        for issue in issues:
            if cnt:  # ETA / progress
                elapsed = time.time() - start
                logger.info(f"[{cnt} / {total_issues}] {round((cnt / total_issues) * 100, 1)}% - ETA {datetime.datetime.fromtimestamp(start + total_issues / (cnt / (elapsed))).strftime('%d %B %Y - %H:%M:%S')}")
            else:
                logger.info(f"[{cnt} / {total_issues}] {round((cnt / total_issues) * 100, 1)}%")

            if cnt % 100 == 0 and cnt:  # write to DB every 100 issues
                logger.info(f"writing up to {len(sims)} rows to DB")
                try:
                    cur.executemany(
                        "INSERT INTO \"%s\" (issue_key, email_id, similarity) VALUES (%%s, %%s, %%s);" % result_db_name,
                        [[s["issue"], s["email"], s["sim"]] for s in sims if s["sim"] != 0])
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error occurred during insert: {str(e)}")
                sims = []

            # calc cosine similarity for current issue and all provided emails
            sims.extend([{
                'issue': issue["key"],
                'email': email["id"],
                'sim': cosine_similarity(
                    first=issue["description_summary_vector"],
                    second=email["body_vector"],
                    doc_occ=doc_dict, 
                    total_docs=total_issues + total_emails)
                } for email in emails])
            cnt += 1
        
        logger.info(f"writing up to {len(sims)} rows to DB")  # write last issues to DB
        cur.executemany("INSERT INTO \"%s\" (issue_key, email_id, similarity) VALUES (%%s, %%s, %%s);" % result_db_name, [[s["issue"], s["email"], s["sim"]] for s in sims if s["sim"] != 0])


def __main__():
    with psycopg.connect(
            host="localhost",
            dbname="relationsDB",
            user="postgres",
            password="UnsavePassword",
            port=5432) as conn:
        cur = conn.cursor(row_factory=psycopg.rows.dict_row)
        # Create tables
        sqlArchEmailsAllIssues = """
        create table if not exists result_arch_emails_all_issues
        (
            email_id integer not null
                constraint "result_arch_emails_all_issues_email_id_fkey"
                    references data_email_email,
            issue_key text not null
                constraint "result_arch_emails_all_issues_issue_key_fkey"
                    references data_jira_jira_issue,
            similarity numeric,
            constraint "result_arch_emails_all_issues_pkey"
                primary key (issue_key, email_id)
        );
        """
        sqlArchIssuesAllEmails = """
                create table if not exists result_arch_issues_all_emails
                (
                    email_id integer not null
                        constraint "result_arch_issues_all_emails_email_id_fkey"
                            references data_email_email,
                    issue_key text not null
                        constraint "result_arch_issues_all_emails_issue_key_fkey"
                            references data_jira_jira_issue,
                    similarity numeric,
                    constraint "result_arch_issues_all_emails_pkey"
                        primary key (issue_key, email_id)
                );
                """
        try:
            cur.execute(sqlArchEmailsAllIssues)
            conn.commit()
            cur.execute(sqlArchIssuesAllEmails)
            conn.commit()
        except Exception as e:
            logger.error(f"Error occurred during creation: {str(e)}")

        cur.execute("SELECT key, description_summary_vector FROM DATA_JIRA_jira_issue;")
        issues = cur.fetchall()

        cur.execute("SELECT id, body_vector FROM DATA_EMAIL_email;")
        emails = cur.fetchall()

        cur.execute("SELECT key, description_summary_vector FROM DATA_JIRA_jira_issue ji WHERE ji.is_design;")
        arch_issues = cur.fetchall()

        cur.execute("select id, body_vector from DATA_EMAIL_email where is_existence or is_executive or is_property;")
        arch_emails = cur.fetchall()
        random.shuffle(arch_issues)  # give all threads equal work
        random.shuffle(issues)

    doc_occ_arch_issues_emails = get_doc_occ(arch_issues, emails)
    doc_occ_issues_arch_emails = get_doc_occ(issues, arch_emails)

    processes = [Process(target=loop, args=(arch_issues[int(len(arch_issues) * (t / THREAD_COUNT)):int(len(arch_issues) * ((t + 1) / THREAD_COUNT))], doc_occ_arch_issues_emails, emails, "result_arch_issues_all_emails")) for t in range(THREAD_COUNT)]
    [p.start() for p in processes]
    [p.join() for p in processes]

    processes = [Process(target=loop, args=(issues[int(len(issues) * (t / THREAD_COUNT)):int(len(issues) * ((t + 1) / THREAD_COUNT))], doc_occ_issues_arch_emails, arch_emails, "result_arch_emails_all_issues")) for t in range(THREAD_COUNT)]
    [p.start() for p in processes]
    [p.join() for p in processes]


if __name__ == "__main__":
    __main__()
