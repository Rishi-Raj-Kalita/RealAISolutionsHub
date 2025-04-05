import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import pandas as pd
import time
from langchain_community.document_loaders import CSVLoader
from pathlib import Path
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

total_owe_previous = (15000 + 5161 + 7742 + 0.18 + 20500)
initial_balance = 1083856.00
current_balance = 1083856.00

categories = [
    'food', 'rent', 'family', 'shopping', 'self-care', 'transport', 'other',
    'unknown'
]
users = ['pallavi', 'prateek', 'aws', 'arshad', 'manas']


class Narration_Type(BaseModel):
    transaction_type: str = Field(
        description=
        "The type of transaction from the list 'food', 'rent', 'family', 'shopping', 'self-care', 'transport', 'other', 'unknown', 'investement"
    )
    mine: float = Field(description="The amount of money spent by the me")
    lent: float = Field(
        description="The amount of money lent by me. If no money lent, then 0")
    lent_by: str = Field(
        description=
        "The person to whom the money is lent from the list 'pallavi', 'prateek', 'aws', 'arshad', 'manas','none'"
    )
    reasoning: str = Field(
        description=
        "The reasoning behind the selecting transaction_type, the amount of money spent by me and lent"
    )


def get_model_rag(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0)
        return llm
    elif (provider == 'aws'):
        from langchain_aws import ChatBedrockConverse
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        llm = ChatBedrockConverse(client=bedrock_client,
                                  model=model,
                                  temperature=0)
        return llm


def get_model(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0)
        structured_llm = llm.with_structured_output(Narration_Type,
                                                    method="json_schema")
        return structured_llm
    elif (provider == 'aws'):
        from langchain_aws import ChatBedrockConverse
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        llm = ChatBedrockConverse(client=bedrock_client,
                                  model=model,
                                  temperature=0)
        structured_llm = llm.with_structured_output(Narration_Type,
                                                    method="json_schema")
        return structured_llm


def get_embeddings(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=model)
        return embeddings
    elif (provider == 'aws'):
        from langchain_aws import BedrockEmbeddings
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        embeddings = BedrockEmbeddings(client=bedrock_client, model_id=model)
        return embeddings


from langchain_core.prompts import PromptTemplate


def classify_narration(narration: str, categories: list, users: list,
                       amount: float):
    # llm=get_model()
    llm = get_model(model='anthropic.claude-3-sonnet-20240229-v1:0',
                    provider='aws')
    category_list = ', '.join(categories)
    users_list = ', '.join(users)
    prompt = PromptTemplate(
        template="""
        'System: You will help me classify my bank transaction into one of the following categories: {categories}.
        Also determine if entire money is spent by me or split between me and someone else. 
        Always remember to add a reasoning for your classification.
        If split, it has to be between following users: {users}'
        Here are some details related to various categories:
        - rent: Includes payments made for rent, which includes my house rent, online subscriptions like netflix, Amazon Prime, jiohotstar. 
        Washing machine rent, Maid rent. Also payment made to airtel, jio and excitel.com for mobile recharge and wifi recharge.
        - food: Includes payment related to food items like cake, pizza, burger. Also purchases from supermarket like Ratnadeep, Tata Star and DMart. Any amount less than 100 is safe to be considered as food if you
        are not able to determine the transaction category. This category will comprise the maximum number of transactions.
        - family: The user will specifically mention family in the transaction. It includes payments made to family members like my father, mother, brother, sister, wife, husband, son, daughter.
        - transport: It includes payments made for fuel, and online transport services like ola and uber.
        - self-care: It includes payments made for personal care like skin-care, hair-cut, gym. Also includes payments for fun activities like movies, comedy shows and concerts.
        - shopping: It includes payments made for shopping like clothes, shoes, bags, accessories, electronics, gadgets, furniture, home decor. The transactions include brands like addidas, nike, zara, h&m, apple, samsung, oneplus, mi, ikea, pepperfry.
        - investment: It includes payments made for investment like mutual funds, stocks, gold, silver, bitcoin, real estate. Also includes payments made for insurance like life insurance, health insurance, car insurance, bike insurance.
        - unknown: If you are unsure of where to put the transaction. Select this category.

        Here are some details related to the users who split money with me:
        - Only if the user says 'by 2' then only split the amount otherwise entire amount is spent by me.
        - All the split transactions will have 'by x' or 'By x' in the narration. This means the money will be equally split by x people.
        - If you are certain that money is lent by me and I have not mentioned any name in the transaction then select 'pallavi'
        - pallavi : My flatmate, will be used in most of the transactions related to rent, food, transport, self-care, shopping.
        - prateek : My other flatmate, rarely used in transactions.
        - aws: My company, which involves transactions related to reimbursements.
        - arshad: My colleague, involes transaction related to food.
        - manas: My other flatmate, rarely used in transactions.
        'User': '{narration}. The total amount spent is {amount}.'
        """,
        input_variables=['narration', 'categories', 'users', 'amount'])
    chain = prompt | llm
    try:
        response = chain.invoke({
            'narration': narration,
            'categories': category_list,
            'users': users_list,
            'amount': amount
        })
        return response
    except Exception as e:
        print("Error:", e)


def get_data(filepath):

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    columns = [
        'Date', 'Narration', 'Value_Date', 'Debit_Amount', 'Credit_Amount',
        'Chq_Ref_Number', 'Closing_Balance'
    ]
    df = pd.read_csv(filepath, names=columns)
    df = df.drop(index=0)
    df = df.drop(columns=['Chq_Ref_Number', 'Value_Date'])

    # Reset the index if needed
    df = df.reset_index(drop=True)
    return df


def traverse_expense(categories: list, users: list, df: pd.DataFrame):
    global current_balance
    global initial_balance
    print("global initial   ", initial_balance)
    print("global current   ", current_balance)

    target_df = pd.DataFrame(columns=[
        'date', 'narration', 'amount', 'mine', 'lent', 'lent_by', 'type',
        'balance', 'closing_balance', 'credit', 'reasoning'
    ])

    for index, row in df.iterrows():
        # print(row['Narration'], row['Debit_Amount'], row['Credit_Amount'])
        # print("current_balance before",current_balance)
        print("-" * 80)
        if float(row['Credit_Amount'].strip()) > 0.0:
            # Create a new row for the target DataFrame
            # print("current_balance before",current_balance)
            amount = float(row['Credit_Amount'].strip())
            current_balance = current_balance + amount
            print(type(amount), type(current_balance),
                  current_balance + amount)
            print(
                f"current_balance after adding amount {amount}= {current_balance}"
            )
            new_row = {
                'date': row['Date'],
                'narration': row['Narration'].lower().strip(),
                'amount': row['Credit_Amount'].strip(),
                'type': 'input_amount',
                'mine': 0,  # Set appropriate value if needed
                'lent': 0,  # Set appropriate value if needed
                'lent_by': 'none',
                'balance': current_balance,  # Set appropriate value if needed
                'closing_balance': row['Closing_Balance'].strip(),
                'credit': 'Y',
                'reasoning': 'none'
            }
            # Append the new row to the target DataFrame
            target_df = pd.concat(
                [target_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            amount = float(row['Debit_Amount'].strip())
            current_balance = current_balance - amount
            print(
                f"current_balance after deducting amount {amount}= {current_balance}"
            )

            response = classify_narration(narration=row['Narration'].lower(),
                                          categories=categories,
                                          users=users,
                                          amount=row['Debit_Amount'])

            transaction_type = response.transaction_type.lower()
            mine = response.mine
            lent = response.lent
            lent_by = response.lent_by
            reasoning = response.reasoning
            print(row['Narration'].strip(), reasoning)

            print("before inserting:", current_balance)
            new_row = {
                'date': row['Date'],
                'narration': row['Narration'].lower().strip(),
                'amount': row['Debit_Amount'].strip(),
                'type': transaction_type,
                'mine': mine,  # Set appropriate value if needed
                'lent': lent,  # Set appropriate value if needed
                'lent_by': lent_by,
                'balance': current_balance,  # Set appropriate value if needed
                'closing_balance': row['Closing_Balance'].strip(),
                'credit': 'N',
                'reasoning': reasoning
            }
            print("new row", new_row)
            # Append the new row to the target DataFrame
            target_df = pd.concat(
                [target_df, pd.DataFrame([new_row])], ignore_index=True)

        print("-" * 80)

    return target_df, current_balance


def get_cleaned_data():
    cleaned_df = pd.read_csv('./data/cleaned/cleaned_df.csv')
    target_df = pd.read_csv('./data/raw/target_df.csv')
    combined_df = pd.concat([cleaned_df, target_df], ignore_index=True)
    combined_df.to_csv('./data/cleaned/cleaned_df.csv', index=False)
    return combined_df


def get_input_amount():
    cleaned_df = pd.read_csv('./data/cleaned/cleaned_df.csv')
    input_df = cleaned_df[cleaned_df['type'] == 'input_amount']
    input_df.to_csv('./data/cleaned/input_df.csv', index=False)
    return input_df


def curate_data() -> pd.DataFrame:
    previous_owe_pallavi = 0
    previous_owe_prateek = (15000 + 20500)
    previous_owe_aws = (5161 + 7742 + 0.18)
    previous_owe_manas = 0
    previous_owe_arshad = 0
    cleaned_df = pd.read_csv('./data/cleaned/cleaned_df.csv')
    input_df = pd.read_csv('./data/cleaned/input_df.csv')
    category_columns = [
        'my_expenses', 'category_food', 'category_rent', 'category_family',
        'category_shopping', 'category_self-care', 'category_transport',
        'category_other', 'catagory_investment', 'user_pallavi',
        'user_prateek', 'user_aws', 'user_arshad', 'user_manas'
    ]
    df = pd.DataFrame(columns=category_columns)

    total_mine_spent = cleaned_df[cleaned_df['mine'] > 0]['mine'].sum()
    total_lent_spent = cleaned_df[cleaned_df['lent'] > 0]['lent'].sum()

    total_food_spent = cleaned_df[cleaned_df['type'] == 'food']['mine'].sum()
    total_rent_spent = cleaned_df[cleaned_df['type'] == 'rent']['mine'].sum()
    total_family_spent = cleaned_df[cleaned_df['type'] ==
                                    'family']['mine'].sum()
    total_shopping_spent = cleaned_df[cleaned_df['type'] ==
                                      'shopping']['mine'].sum()
    total_self_care_spent = cleaned_df[cleaned_df['type'] ==
                                       'self-care']['mine'].sum()
    total_transport_spent = cleaned_df[cleaned_df['type'] ==
                                       'transport']['mine'].sum()
    total_other_spent = cleaned_df[cleaned_df['type'] == 'other']['mine'].sum()
    total_investment_spent = cleaned_df[cleaned_df['type'] ==
                                        'investment']['mine'].sum()

    total_paid_pallavi = input_df[input_df['lent_by'] ==
                                  'pallavi']['amount'].sum()
    total_paid_prateek = input_df[input_df['lent_by'] ==
                                  'prateek']['amount'].sum()
    total_paid_aws = input_df[input_df['lent_by'] == 'aws']['amount'].sum()
    total_paid_arshad = input_df[input_df['lent_by'] ==
                                 'arshad']['amount'].sum()
    total_paid_manas = input_df[input_df['lent_by'] == 'manas']['amount'].sum()

    total_owe_pallavi = cleaned_df[cleaned_df['lent_by'] ==
                                   'pallavi']['lent'].sum()
    total_owe_prateek = cleaned_df[cleaned_df['lent_by'] ==
                                   'prateek']['lent'].sum()
    total_owe_aws = cleaned_df[cleaned_df['lent_by'] == 'aws']['lent'].sum()
    total_owe_arshad = cleaned_df[cleaned_df['lent_by'] ==
                                  'arshad']['lent'].sum()
    total_owe_manas = cleaned_df[cleaned_df['lent_by'] ==
                                 'manas']['lent'].sum()

    df.loc[0, 'my_expenses'] = total_mine_spent

    df.loc[0, 'category_food'] = total_food_spent
    df.loc[0, 'category_rent'] = total_rent_spent
    df.loc[0, 'category_family'] = total_family_spent
    df.loc[0, 'category_shopping'] = total_shopping_spent
    df.loc[0, 'category_self-care'] = total_self_care_spent
    df.loc[0, 'category_transport'] = total_transport_spent
    df.loc[0, 'category_other'] = total_other_spent
    df.loc[0, 'catagory_investment'] = total_investment_spent
    df.loc[0, 'user_pallavi'] = (total_owe_pallavi +
                                 previous_owe_pallavi) - total_paid_pallavi
    df.loc[0, 'user_prateek'] = (total_owe_prateek +
                                 previous_owe_prateek) - total_paid_prateek
    df.loc[0, 'user_aws'] = (total_owe_aws + previous_owe_aws) - total_paid_aws
    df.loc[0, 'user_arshad'] = (total_owe_arshad +
                                previous_owe_arshad) - total_paid_arshad
    df.loc[0, 'user_manas'] = (total_owe_manas +
                               previous_owe_manas) - total_paid_manas
    df.loc[0, 'lent'] = df.loc[0, 'user_pallavi'] + df.loc[
        0, 'user_prateek'] + df.loc[0, 'user_aws'] + df.loc[0, 'user_arshad']

    df.to_csv('./data/curated/curated_df.csv', index=False)

    return df


def check_balance(current_balance):
    global total_owe_previous
    df_curated = pd.read_csv('./data/curated/curated_df.csv')
    mine = df_curated['my_expenses'].values[0]
    lent = df_curated['user_pallavi'].values[0] + df_curated[
        'user_prateek'].values[0] + df_curated['user_aws'].values[
            0] + df_curated['user_arshad'].values[0]
    global initial_balance

    print("Initial Balance:", initial_balance)
    print("Current Balance:", current_balance)
    print("My Expenses:", mine)
    print("Lent:", lent)

    diff = current_balance - (initial_balance - mine -
                              lent) - total_owe_previous

    if (diff < 500.00):
        print("The balance is correct")
        return f"""The balance is correct.\n
        Initial Balance:{initial_balance}\n
        Current Balance:{current_balance}\n
        My Expense:{mine}
        Money Lent:{lent}
        """
    else:
        print("The balance is incorrect by amount:",
              current_balance - (initial_balance - mine - lent))
        return f"The balance is incorrect by amount: {diff}"


def extract_owe():
    cleaned_df = pd.read_csv('./data/cleaned/cleaned_df.csv')
    df_pallavi = cleaned_df[cleaned_df['lent_by'] == 'pallavi']
    df_prateek = cleaned_df[cleaned_df['lent_by'] == 'prateek']
    df_aws = cleaned_df[cleaned_df['lent_by'] == 'aws']
    df_arshad = cleaned_df[cleaned_df['lent_by'] == 'arshad']

    # Define the columns for the lent_df DataFrame
    columns = ['person', 'narration', 'amount', 'date']

    # Create an empty DataFrame with the specified columns
    lent_df = pd.DataFrame(columns=columns)

    # Function to append data to lent_df
    def append_to_lent_df(df, person):
        for index, row in df.iterrows():
            lent_df.loc[len(lent_df)] = [
                person, row['narration'], row['lent'], row['date']
            ]

    # Append data for each person
    append_to_lent_df(df_pallavi, 'Pallavi')
    append_to_lent_df(df_prateek, 'Prateek')
    append_to_lent_df(df_aws, 'AWS')
    append_to_lent_df(df_arshad, 'Arshad')

    return lent_df


def get_data_rag():
    cleaned_df = pd.read_csv('./data/cleaned/cleaned_df.csv')
    rag_df = cleaned_df[['amount', 'type', 'date', 'narration']]
    return rag_df


def rag_expense():
    llm = get_model_rag()
    # llm = get_model_rag(model='anthropic.claude-3-sonnet-20240229-v1:0',
    # provider='aws')

    rag_df = get_data_rag()
    file_path = './data/curated/rag_df.csv'
    rag_df.to_csv(file_path, index=False)
    loader = CSVLoader(file_path=file_path)
    docs = loader.load_and_split()
    embeddings = get_embeddings()
    # embeddings=get_embeddings(model='amazon.titan-embed-text-v2:0', provider='aws')
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(embedding_function=embeddings,
                         index=index,
                         docstore=InMemoryDocstore(),
                         index_to_docstore_id={})
    vector_store.add_documents(documents=docs)
    retriever = vector_store.as_retriever()

    # Set up system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain
