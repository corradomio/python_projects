import argparse
import openai
import re
import time

client = openai.OpenAI()


def create_prompt(question):
    ''' Generate prompt to translate question into SQL query.
    Args:
    question: question about data in natural language.
    Returns:
    prompt for question translation.
    '''
    parts = []
    parts += ['Database:']
    parts += ['create table games(rank int, name text, platform text,']
    parts += ['year int, genre text, publisher text, americasales numeric,']
    parts += ['eusales numeric, japansales numeric, othersales numeric,']
    parts += ['globalsales numeric);']
    parts += ['Translate this question into SQL query:']
    parts += [question]
    return '\n'.join(parts)


def call_llm(prompt):
    '''
    Args:
    prompt: input prompt for language model.
    Returns:
    Answer by language model.
    '''
    for nr_retries in range(1, 4):
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {'role':'user', 'content':prompt}
                ]
            )
            return response.choices[0].message.content
        except:
            time.sleep(nr_retries * 2)
            raise Exception('Cannot query OpenAI model!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str, help='A question about games')
    args = parser.parse_args()
    prompt = create_prompt(args.question)
    answer = call_llm(prompt)
    print(f'Answer: {answer}')
    query = re.findall('```sql(.*)```', answer, re.DOTALL)[0]
    print(f'SQL: {query}')
