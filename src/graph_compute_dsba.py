import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
import graph_analysis as g
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Running on whole time and merging to big df...")
metrics_full = {}
df_pop, eu_iso = g.load_population_df()
big_df_col = ["year","month","prod","country"] +  ['edges','nodes','size','degree',
    'degree_perc',
    'out_degree',
    'out_degree_perc',
    'in_degree',
    'in_degree_perc',
    'density',
    'density_adj',
    'clustering',
    'page_rank',
    'vulnerability',
    'in_weight_degree',
    'out_weight_degree',
    'degree_gamma',
    'degree_xmin',
    'in_degree_gamma',
    'in_degree_xmin',
    'out_degree_gamma',
    'out_degree_xmin',
    'in_weight_degree_gamma',
    'in_weight_degree_xmin',
    'out_weight_degree_gamma',
    'out_weight_degree_xmin',
    'out_weight_abs',
    'in_weight_abs']

big_df = pd.DataFrame(columns=big_df_col)
issues = []
for y in tqdm([y for y in range(2001,2021)]):
    df = g.load_filtered_data("complete", save=True, force_reload=True, verbose=False, sorted=False, columns=g.full_columns, types=g.full_types, group_by_prod_code=True, n_digits=2,
                                years=[y], months=None, product="all", flow="all", declarant_iso="all", partner_iso="all", trade_type="all")
    n4g = g.get_world_countries(df_pop, eu_iso, year=str(y))
    for prod in tqdm(df.PRODUCT_CPA2_1.unique(),leave=False):
        df4g = df[df.PRODUCT_CPA2_1 == prod]
        t4g = g.extract_table_for_graph(df4g, y=str(y), flow="all", scale_by='population', pop_df=df_pop)
        if t4g.shape[0] < 10:
            issues.append((y,prod,t4g.shape[0]))
        coord, metrics, G = g.makeGraph(t4g, tab_nodes=None, weight_flag=True, criterio="VALUE_IN_EUROS")
        nx.write_gexf(G,f"data-samples/graphs/complete/complete_y{y}_p{prod}.gexf")
        x = metrics.reset_index().to_dict()
        x["year"] = str(y)
        x["month"] = "-"
        x["prod"] = prod
        big_df = pd.concat([big_df,pd.DataFrame(x,columns=big_df_col)],axis=0)
        metrics_full[f"{y}_cpa_{prod}"] = metrics

with open("./data-samples/metrics/issues_complete.txt","w") as f:
    f.write(str(issues))
pickle.dump(metrics_full,open("./data-samples/manual/metrics/metrics_complete_V_2.pickle","wb"))
big_df.to_parquet("./data-samples/manual/metrics/metrics_complete_V_2.parquet")
