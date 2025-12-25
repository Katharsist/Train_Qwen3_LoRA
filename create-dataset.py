import pandas as pd

df = pd.read_csv('lenta-ru-news.csv', low_memory=False)

politic_mask = df['tags'] == 'Политика'
politic_rows = df[politic_mask]
print(f"[*] Всего новостей: {len(df)}")
print(f"[*] Новости с тегом 'Политика': {len(politic_rows)}")

filtered_rows = []
for index, row in politic_rows.iterrows():
    all_fields_ok = True
    for value in row:
        if pd.isna(value):
            all_fields_ok = False
            break
    if all_fields_ok:
        filtered_rows.append(row)

filtered_news = pd.DataFrame(filtered_rows)
for column in filtered_news.columns:
    filtered_news[column] = filtered_news[column].astype(str).str.replace('\xa0', ' ')

filtered_news.to_csv('lenta-ru-news-politics.csv', index=False)
print(f"[*] Отфильтрованные новости с тегом 'Политика': {len(filtered_news)}")