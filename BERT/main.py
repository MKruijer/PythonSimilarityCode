import psycopg
import time
import datetime
import logging
import sys
import pickle
from multiprocessing import Process
from sentence_transformers import SentenceTransformer, util
import psutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(f'[%(threadName)s] [%(asctime)s] [%(levelname)s] --- %(message)s')
stdouthandler = logging.StreamHandler(sys.stdout)
stdouthandler.setLevel(logging.INFO)
stdouthandler.setFormatter(formatter)
logger.addHandler(stdouthandler)
ITERATION = "iter4"


def get_all_jiras(cursor):
    query = """
        SELECT key, concat(description, ' ', summary) as body
        FROM data_jira_jira_issue
    """
    cursor.execute(query)
    issues = cursor.fetchall()
    return issues


def get_all_emails(cursor):
    query = """
        SELECT id, body
        FROM data_email_email
    """
    cursor.execute(query)
    emails = cursor.fetchall()
    return emails


def get_arch_jiras(cursor):
    query = """
        SELECT key, concat(description, ' ', summary) as body
        FROM data_jira_jira_issue
        WHERE is_design
    """
    cursor.execute(query)
    issues = cursor.fetchall()
    return issues


def get_arch_emails(cursor):
    query = """
    SELECT e.id, e.body
    FROM data_email_email e
    WHERE e.id IN (
        SELECT et.email_id
        FROM DATA_EMAIL_email_tag et
        LEFT JOIN DATA_EMAIL_tag t ON et.tag_id = t.id
        WHERE t.architectural
    )
    """
    cursor.execute(query)
    emails = cursor.fetchall()
    return emails


def calculate_similarity(issues_dict, emails_dict, result_db_name):
    start = time.time()
    with psycopg.connect(
            host="localhost",
            dbname="relationsDB",
            user="postgres",
            password="UnsavePassword",
            port=5432) as connection:
        cursor = connection.cursor(row_factory=psycopg.rows.dict_row)
        total_issues = len(issues_dict)

        cnt = 0
        similarity_list = []
        for issue in issues_dict:
            if cnt:  # ETA / progress
                elapsed = time.time() - start
                logger.info(
                    f"[{cnt} / {total_issues}] {round((cnt / total_issues) * 100, 1)}% - ETA {datetime.datetime.fromtimestamp(start + total_issues / (cnt / elapsed)).strftime('%d %B %Y - %H:%M:%S')}")
            else:
                logger.info(f"[{cnt} / {total_issues}] {round((cnt / total_issues) * 100, 1)}%")

            if cnt % 30 == 0 and cnt:  # write to DB every 30 issues
                values = [(s['issue'], s['email'], s['similarity']) for s in similarity_list if s['similarity'] > 0]
                logger.info(f"writing up to {len(values)} rows to DB")
                sql = f"INSERT INTO {result_db_name} (issue_key, email_id, similarity) VALUES (%s, %s, %s)"
                try:
                    cursor.executemany(sql, values)
                    connection.commit()
                except Exception as e:
                    logger.error(f"Error occurred during insert: {str(e)}")
                similarity_list = []
            for email in emails_dict:
                arch_embedding = issues_dict[issue]
                all_embedding = emails_dict[email]
                cosine_sim = util.cos_sim(arch_embedding, all_embedding)
                similarity_list.extend([
                    {'issue': issue, 'email': email, 'similarity': cosine_sim.item()}
                ])
            cnt += 1
        values = [(s['issue'], s['email'], s['similarity']) for s in similarity_list if s['similarity'] > 0]
        logger.info(f"writing up to {len(values)} rows to DB")
        sql = f"INSERT INTO {result_db_name} (issue_key, email_id, similarity) VALUES (%s, %s, %s)"
        try:
            cursor.executemany(sql, values)
            connection.commit()
        except Exception as e:
            logger.error(f"Error occurred during insert: {str(e)}")
    connection.close()


def save_email_or_issue_embedding(list_of_source, source_uid_string, model, save_name):
    source_bodies = [item['body'] for item in list_of_source]
    source_ids = [item[source_uid_string] for item in list_of_source]
    source_embeddings = model.encode(source_bodies)
    # Store sentences & embeddings on disc
    with open(f'{save_name}.pkl', "wb") as fOut:
        pickle.dump({'ids': source_ids, 'embeddings': source_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_email_or_issue_embedding(save_name):
    with open(f'{save_name}.pkl', "rb") as fIn:
        data = pickle.load(fIn)

    source_ids = data['ids']
    source_embeddings = data['embeddings']
    result_dict = {id_val: embedding for id_val, embedding in zip(source_ids, source_embeddings)}
    return result_dict


def split_dictionary(dictionary, amount_of_splits):
    keys = list(dictionary.keys())
    split_size = len(keys) // amount_of_splits
    split_dictionaries = []

    for i in range(amount_of_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size
        split_keys = keys[start_index:end_index]
        split_dict = {key: dictionary[key] for key in split_keys}
        split_dictionaries.append(split_dict)

    return split_dictionaries


def __main__():
    # Get the total number of logical CPUs
    cpu_count = psutil.cpu_count(logical=True)
    desired_cpu_limit = 0.8  # 80% CPU usage limit
    thread_limit = int(cpu_count * desired_cpu_limit)

    # Limit the CPU affinity of the current process
    psutil.Process().cpu_affinity(list(range(thread_limit)))
    with psycopg.connect(
            host="localhost",
            dbname="relationsDB",
            user="postgres",
            password="UnsavePassword",
            port=5432) as connection:
        cur = connection.cursor(row_factory=psycopg.rows.dict_row)
        all_issues = get_all_jiras(cur)
        all_emails = get_all_emails(cur)
        arch_emails = get_arch_emails(cur)
        arch_issues = get_arch_jiras(cur)

        # Create database tables
        sql_arch_email_all_issue = f"""
        create table if not exists {ITERATION}_sim_result_arch_emails_all_issues
        (
            email_id integer not null
                constraint "{ITERATION}_SIM_RESULT_arch_emails_all_issues_email_id_fkey"
                    references data_email_email,
            issue_key text not null
                constraint "{ITERATION}_SIM_RESULT_arch_emails_all_issues_issue_key_fkey"
                    references data_jira_jira_issue,
            similarity numeric,
            constraint "{ITERATION}_SIM_RESULT_arch_emails_all_issues_pkey"
                primary key (issue_key, email_id)
        );
        """
        sql_arch_issue_all_email = f"""
                create table if not exists {ITERATION}_sim_result_arch_issues_all_emails
                (
                    email_id integer not null
                        constraint "{ITERATION}_SIM_RESULT_arch_issues_all_emails_email_id_fkey"
                            references data_email_email,
                    issue_key text not null
                        constraint "{ITERATION}_SIM_RESULT_arch_issues_all_emails_issue_key_fkey"
                            references data_jira_jira_issue,
                    similarity numeric,
                    constraint "{ITERATION}_SIM_RESULT_arch_issues_all_emails_pkey"
                        primary key (issue_key, email_id)
                );
                """
        cur.execute(sql_arch_email_all_issue)
        cur.execute(sql_arch_issue_all_email)

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    logger.info(f"loading model {model_name}")
    model = SentenceTransformer(model_name, device='cuda')
    logger.info(f"loaded model {model_name}")

    # Saving models in case of future need
    logger.info(f"saving arch_emails_dictionary")
    save_email_or_issue_embedding(arch_emails, 'id', model, 'models/arch-emails-embeddings')
    logger.info(f"saving arch_issues_dictionary")
    save_email_or_issue_embedding(arch_issues, 'key', model, 'models/arch-issues-embeddings')
    logger.info(f"saving all_emails_dictionary")
    save_email_or_issue_embedding(all_emails, 'id', model, 'models/all-emails-embeddings')
    logger.info(f"saving all_emails_dictionary")
    save_email_or_issue_embedding(all_issues, 'key', model, 'models/all-issues-embeddings')
    logger.info(f"done saving all dictionaries.")

    # Load models
    arch_emails_dict = load_email_or_issue_embedding('models/arch-emails-embeddings')
    arch_issues_dict = split_dictionary(load_email_or_issue_embedding('models/arch-issues-embeddings'), thread_limit)
    all_emails_dict = load_email_or_issue_embedding('models/all-emails-embeddings')
    all_issues_dict = split_dictionary(load_email_or_issue_embedding('models/all-issues-embeddings'), thread_limit)

    # Start calculating the similarity scores and insert them in the database
    processes = [Process(target=calculate_similarity, args=(
        arch_issues_dict[t],
        all_emails_dict,
        f"{ITERATION}_sim_result_arch_issues_all_emails  ")) for t in range(thread_limit)]
    [p.start() for p in processes]
    [p.join() for p in processes]

    processes = [Process(target=calculate_similarity, args=(
        all_issues_dict[t],
        arch_emails_dict,
        f"{ITERATION}_sim_result_arch_emails_all_issues")) for t in range(thread_limit)]
    [p.start() for p in processes]
    [p.join() for p in processes]


if __name__ == "__main__":
    __main__()
