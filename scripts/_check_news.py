"""Check news data for 20240619."""
import pandas as pd

df = pd.read_excel("data/external/news/2024_News_Security.xlsx", skiprows=2)
df.columns = ["NewsID", "DeclareDate", "Title", "Symbol", "ShortName",
              "SecurityTypeID", "SecurityType", "FullDeclareDate"]
df["DeclareDate"] = pd.to_datetime(df["DeclareDate"], errors="coerce")

print("Total news records: {}".format(len(df)))
print("Date range: {} to {}".format(df["DeclareDate"].min(), df["DeclareDate"].max()))
print()

# 20240619
day = df[df["DeclareDate"] == "2024-06-19"]
n_stocks = day["Symbol"].nunique()
print("20240619: {} news, {} stocks".format(len(day), n_stocks))
print()

print("Sample news:")
for _, r in day.head(8).iterrows():
    sym = r["Symbol"]
    name = r["ShortName"]
    title = str(r["Title"])[:60]
    print("  [{}] {}: {}".format(sym, name, title))

print()
print("Daily counts:")
for d in ["2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21"]:
    n = len(df[df["DeclareDate"] == d])
    print("  {}: {} news".format(d, n))

# Check A-share stocks overlap
a_share = day[day["Symbol"].str.match(r"^\d{6}$", na=False)]
print()
print("A-share news on 20240619: {} records, {} stocks".format(len(a_share), a_share["Symbol"].nunique()))
