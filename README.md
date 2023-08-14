# Cosine similarity code
These 2 python script exist to apply cosine similarity between issues and emails.

## SBERT
Here we create 2 tables that contain the cosine similarity values. At the top of the file is a variable to keep track of the iterations. We use the model with the name 'sentence-transformers/all-MiniLM-L6-v2', which can be changed in the setup_models() function.

### How to run
Change the iteration variable to the desired number. Update the psycopg.connect variables to match with your database. In the main we set a thread limit, which by default is 80% but can be changed by changing desired_cpi_limit. Optionally you can change where the embeddings are saved.

## JulianCosine

The JulianCosine is a TF-IDF based cosine similarity. We did not create this and all credit goes to the authors, whose paper can be found Here: https://julianpasveer.com/Relate_architectural_issues_and_emails_in_mailing_lists%20(2).pdf