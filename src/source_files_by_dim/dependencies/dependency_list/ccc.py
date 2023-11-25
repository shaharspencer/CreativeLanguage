import pandas as pd

c = pd.read_csv("old_watch/watch_clause_example_dobjs.csv", encoding='ISO-8859-1')

def my_function(st):
  st = st.strip()
  if st[-1] in "aeiou":
    return f"somebody watched an {st}"
  return f"somebody watched a {st}"

c['prompt'] = c['dobj'].apply(my_function)

c.to_csv("prompt for dobjs.csv", encoding='utf-8')