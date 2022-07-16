import hashlib, random, math, os, powerlaw
import pandas as pd
import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

full_columns = ['PRODUCT_CPA2_1', 'DECLARANT_ISO', 'PARTNER_ISO', 'FLOW', 'PERIOD', 'VALUE_IN_EUROS', 'QUANTITY_IN_KG']
full_types = {'PRODUCT_CPA2_1':str, 'DECLARANT_ISO':str, 'PARTNER_ISO':str, 'FLOW':int, 'PERIOD':str, 'VALUE_IN_EUROS':np.int64, 'QUANTITY_IN_KG':np.int64}

wto_columns = ['PRODUCT_CPA2_1', 'DECLARANT_ISO', 'PARTNER_ISO', 'FLOW', 'PERIOD', 'VALUE_IN_EUROS', 'QUANTITY_IN_KG']
wto_types = {'PRODUCT_CPA2_1':str, 'DECLARANT_ISO':str, 'PARTNER_ISO':str, 'FLOW':int, 'PERIOD':str, 'VALUE_IN_EUROS':np.int64, 'QUANTITY_IN_KG':np.int64}

tr_columns = ['PRODUCT_NSTR', 'DECLARANT_ISO', 'PARTNER_ISO', 'FLOW', 'PERIOD', 'TRANSPORT_MODE', 'VALUE_IN_EUROS']
tr_types = {'PRODUCT_NSTR':str, 'DECLARANT_ISO':str, 'PARTNER_ISO':str, 'FLOW':int, 'PERIOD':str, 'TRANSPORT_MODE':int, 'VALUE_IN_EUROS':np.int64}

tr_intra_columns = ['PRODUCT_NSTR', 'DECLARANT_ISO', 'PARTNER_ISO', 'FLOW', 'PERIOD', 'TRANSPORT_MODE', 'VALUE_IN_EUROS']
tr_intra_types = {'PRODUCT_NSTR':str, 'DECLARANT_ISO':str, 'PARTNER_ISO':str, 'FLOW':int, 'PERIOD':str, 'TRANSPORT_MODE':int, 'VALUE_IN_EUROS':np.int64}

files_dict = {"full":pd.read_table("./data-samples/nomenclature/CPA2_1.txt",header=None).rename({0:"code",1:"name"},axis=1),
                "tr":pd.read_table("./data-samples/nomenclature/NSTR.txt",header=None).rename({0:"code",1:"name"},axis=1),
                "tr_intra_":pd.read_table("./data-samples/nomenclature/NSTR.txt",header=None).rename({0:"code",1:"name"},axis=1)}

def combine_sources(df_com, df_wto):
    df_extra = df_wto[~(df_wto.DECLARANT_ISO.isin(df_com.DECLARANT_ISO.unique()))&~(df_wto.PARTNER_ISO.isin(df_com.DECLARANT_ISO.unique()))]
    return pd.concat([df_com,df_extra],axis=0).reset_index(drop=True)

def load_filtered_data(table, save=False, verbose=1, force_reload=False, sorted=True, **params):
    """"
    load_filtered_data("full", save=True, force_reload=True, columns=g.full_columns, types=g.full_types, 
                        group_by_prod_code=True, n_digits=2, years=[2012], months=None, product="all", 
                        flow="all", trade_type="all",  declarant_iso="all", partner_iso="all")

    table: str
    save: bool
    types: dict(type)
    columns: list
    group_by_prod_code: bool 
    n_digits: int
    years: list(int)
    months: list(int) | None | "all"
    product: "all" | list(str) 
    flow: 1 | 2 | "all"
    trade_type: "I" | "E" | "all"
    declarant_iso: str | "all"
    partner_iso: str | "all"

    Returns
        (pd.DataFrame, str)
    """
    
    if verbose:
        print("loading",table,end=": ")

    def get_params_str_hash(params):
        str_params = {x:params[x] for x in params if x != "types"}
        # str_params = params
        # return str(str_params).replace("{","").replace("}","").replace("': ","_").replace("[","").replace("]","").replace(", ","_").replace("'","").replace("<class ","").replace(">","")
        return hashlib.sha256(str(str_params).encode("UTF_8")).hexdigest()

    def group_by_prod_code(table,**params):
        df = table.copy()
        column = df.columns[0]
        df[column] = df[column].str[:params["n_digits"]]
        return df.groupby(params["columns"][:-2],as_index=False).sum()
    
    if not force_reload and os.path.exists(f"./data-samples/manual/{table}/{table}__{get_params_str_hash(params)}.parquet"):
        if not os.path.exists(f"./data-samples/manual/{table}/"):
            os.makedirs(f"./data-samples/manual/{table}/")
        if verbose:
            print("loading existing...",end=" ")
        df = pd.read_parquet(f"./data-samples/manual/{table}/{table}__{get_params_str_hash(params)}.parquet")
        if verbose:
            print("Table loaded")
        return df, None
    else:
        prod_code_df = params["columns"][0]

        def build_filters(df,params,table,name,prod_code_df):
            filters = pd.Series([True for i in range(df.shape[0])])
            if isinstance(params["product"],list):
                filter_prod = ~filters.copy()
                not_f = []
                for p in params["product"]:
                    mask = (df[prod_code_df] == p)
                    if not mask.any():
                        not_f.append(p)
                    filter_prod = filter_prod | mask
                if verbose==2 and len(not_f) > 0:
                    print(','.join(not_f),f"not found in {table}{name}",end=", ")
                filters = filters & filter_prod
            elif params["product"] != "all":
                mask = (df[prod_code_df] == params["product"])
                if verbose==2 and not mask.any():
                    print(f"{params['product']} not found in {table}{name}",end=", ")
                filters = filters & mask
            if params["flow"] != "all":
                filters = filters & (df.FLOW == params["flow"])
            if table == "full" and params["trade_type"] != "all":
                filters = filters & (df.TRADE_TYPE == params["trade_type"])
            if isinstance(params["declarant_iso"],list):
                filter_dec = ~filters.copy()
                for p in params["declarant_iso"]:
                    mask = (df["DECLARANT_ISO"] == p)
                    if verbose==2 and not mask.any():
                        print(f"{p} not found in {table}{name}",end=", ")
                    filter_dec = filter_dec | mask
                filters = filters & filter_dec
            elif params["declarant_iso"] != "all":
                mask = (df["DECLARANT_ISO"] == params["declarant_iso"])
                if verbose==2 and not mask.any():
                    print(f"{params['declarant_iso']} not found in {table}{name}",end=", ")
            if isinstance(params["partner_iso"],list):
                filter_dec = ~filters.copy()
                for p in params["partner_iso"]:
                    mask = (df["PARTNER_ISO"] == p)
                    if verbose==2 and not mask.any():
                        print(f"{p} not found in {table}{name}",end=", ")
                    filter_dec = filter_dec | mask
                filters = filters & filter_dec
            elif params["partner_iso"] != "all":
                mask = (df["PARTNER_ISO"] == params["partner_iso"])
                if verbose==2 and not mask.any():
                    print(f"{params['partner_iso']} not found in {table}{name}",end=", ")
            
            if verbose==2 and not filters.any():
                print(f"\n\tempty df for {table}{name}")

            return filters

        if table not in ["full","tr","tr_intra_","wto","complete"]:
            raise Exception("table not valid")
        df = pd.DataFrame(columns=params["columns"])
        for y in tqdm(params["years"],leave=False):
            if verbose:
                print(y,end=" ")
            if table == "complete":
                df_com = pd.read_parquet(f"./data-samples/full/Years/full{y}52.parquet",columns=params["columns"])
                df_wto = pd.read_parquet(f"./data-samples/wto/Years/wto{y}52.parquet",columns=params["columns"])
                df_y = combine_sources(df_com,df_wto)
                if "group_by_prod_code" in params.keys() and params["group_by_prod_code"]:
                    df_y_filtered = group_by_prod_code(df_y[build_filters(df_y,params,table,str(y)+"52",prod_code_df)].astype(params["types"]),**params)
                else:
                    df_y_filtered = df_y[build_filters(df_y,params,table,str(y)+"52",prod_code_df)].astype(params["types"])
                df = pd.concat([df,df_y_filtered])
            else:
                if params["months"] is None:
                    df_y = pd.read_parquet(f"./data-samples/{table}/Years/{table}{y}52.parquet",columns=params["columns"])
                    # print("df_y",(df_y.VALUE_IN_EUROS < 0).any())
                    if "group_by_prod_code" in params.keys() and params["group_by_prod_code"]:
                        df_y_filtered = group_by_prod_code(df_y[build_filters(df_y,params,table,str(y)+"52",prod_code_df)].astype(params["types"]),**params)
                    else:
                        df_y_f = df_y[build_filters(df_y,params,table,str(y)+"52",prod_code_df)]
                        # print("df_y_f",(df_y_f.VALUE_IN_EUROS < 0).any())
                        df_y_filtered = df_y_f.astype(params["types"])
                    # print("df_y_filtered",(df_y_filtered.VALUE_IN_EUROS < 0).any())
                    df = pd.concat([df,df_y_filtered])
                    # print("df",(df.VALUE_IN_EUROS < 0).any())
                    del df_y, df_y_f, df_y_filtered
                elif params["months"] == "all":
                    for m in tqdm(range(1,13),leave=False):
                        # if verbose:
                        #     print(str(m),end=" ")
                        df_ym = pd.read_parquet(f"./data-samples/{table}/Months/{table}{y*100+m}.parquet",columns=params["columns"])
                        if "group_by_prod_code" in params.keys() and params["group_by_prod_code"]:
                            df_y_filtered = group_by_prod_code(df_ym[build_filters(df_ym,params,table,str(y*100+m),prod_code_df)].astype(params["types"]),**params)
                        else:
                            df_y_filtered = df_ym[build_filters(df_ym,params,table,str(y*100+m),prod_code_df)].astype(params["types"])
                        df = pd.concat([df,df_y_filtered])
                    del df_ym, df_y_filtered
                else:
                    for m in params["months"]:
                        if verbose:
                            print(str(m),end=" ")
                        df_ym = pd.read_parquet(f"./data-samples/{table}/Months/{table}{y*100+m}.parquet",columns=params["columns"])
                        if "group_by_prod_code" in params.keys() and params["group_by_prod_code"]:
                            df_y_filtered = group_by_prod_code(df_ym[build_filters(df_ym,params,table,str(y*100+m),prod_code_df)].astype(params["types"]),**params)
                        else:
                            df_y_filtered = df_ym[build_filters(df_ym,params,table,str(y*100+m),prod_code_df)].astype(params["types"])
                        df = pd.concat([df,df_y_filtered])
                    del df_ym, df_y_filtered
        if df.shape[0] == 0:
            if verbose:
                print("\n\tEmpty table")
            return df
        else:
            # print("df",(df.VALUE_IN_EUROS < 0).any())
            if sorted:
                df_out = df.groupby(params["columns"][:-2],as_index=False).sum().sort_values(params["columns"][-2],ascending=False).reset_index(drop=True)
            else:
                df_group = df.groupby(params["columns"][:-2],as_index=False).sum()
                # print("df_group",(df_group.VALUE_IN_EUROS < 0).any())
                df_out = df_group.reset_index(drop=True)
            # print("df_out",(df_out.VALUE_IN_EUROS < 0).any())
            str_save = f"./data-samples/manual/{table}/{table}__{get_params_str_hash(params)}.parquet"
            # return df_out
            df_final = df_out.astype(params["types"])
            # print("df_final",(df_final.VALUE_IN_EUROS < 0).any())
            # return df_y, df_y_filtered, df, df_out, df_final
            if save:
                try:
                    df_out.to_parquet(str_save)
                    if verbose:
                        print(",\t Table loaded")
                except:
                    if verbose:
                        print("\nFailed saving",end="")
                        print(", Table loaded")
                return df_final
            else:
                return df_final                    

def combine_flows(df_to_combine, columns):

    # DEPRECATED
    # df_to_combine.sort_values(["TRADE_TYPE","FLOW"],inplace=True)
    # df_to_combine.reset_index(drop=True,inplace=True)
    # extra_import = df_to_combine[(df_to_combine.TRADE_TYPE == "E")&(df_to_combine.FLOW == 1)]
    # extra_export = df_to_combine[(df_to_combine.TRADE_TYPE == "E")&(df_to_combine.FLOW == 2)]
    # intra_import = df_to_combine[(df_to_combine.TRADE_TYPE == "I")&(df_to_combine.FLOW == 1)]
    # intra_export = df_to_combine[(df_to_combine.TRADE_TYPE == "I")&(df_to_combine.FLOW == 2)]
    # df_to_combine["country_to"] = pd.concat([extra_import['DECLARANT_ISO'],extra_export['PARTNER_ISO'],intra_import["DECLARANT_ISO"],intra_export["PARTNER_ISO"]])
    # df_to_combine["country_from"] = pd.concat([extra_import['PARTNER_ISO'],extra_export['DECLARANT_ISO'],intra_import["PARTNER_ISO"],intra_export["DECLARANT_ISO"]])
    # df_to_scale = df_to_combine.groupby(['TRADE_TYPE', 'PERIOD', 'country_from', 'country_to'],as_index=False)["VALUE_IN_EUROS"].mean()

    df4g = df_to_combine.copy()
    df_eu = df4g[df4g.PARTNER_ISO.isin(df4g.DECLARANT_ISO.unique())]
    df_ex = df4g[~df4g.PARTNER_ISO.isin(df4g.DECLARANT_ISO.unique())]
    # df_eu
    df_to_avg = df_eu[df_eu.FLOW == 1].merge(df_eu[df_eu.FLOW ==2][["DECLARANT_ISO","PARTNER_ISO","VALUE_IN_EUROS","QUANTITY_IN_KG"]]
                                            ,left_on=["DECLARANT_ISO","PARTNER_ISO"]
                                            ,right_on=["PARTNER_ISO","DECLARANT_ISO"]
                                            ,how="inner",suffixes=("","_y"))
    df_to_avg["VALUE_IN_EUROS"] = (df_to_avg["VALUE_IN_EUROS"] + df_to_avg["VALUE_IN_EUROS_y"])/2
    df_to_avg["QUANTITY_IN_KG"] = (df_to_avg["QUANTITY_IN_KG"] + df_to_avg["QUANTITY_IN_KG_y"])/2
    df_eu_l = [df_eu[df_eu.FLOW == 1].merge(df_eu[df_eu.FLOW ==2][["DECLARANT_ISO","PARTNER_ISO","VALUE_IN_EUROS","QUANTITY_IN_KG"]]
                                            ,left_on=["DECLARANT_ISO","PARTNER_ISO"]
                                            ,right_on=["PARTNER_ISO","DECLARANT_ISO"]
                                            ,how="outer",suffixes=("","_y")
                                            ,indicator=True).query('_merge=="left_only"')[columns]
            ,df_to_avg[columns]
            ,df_ex[df_ex.FLOW == 1]
            ,df_ex[df_ex.FLOW == 2].rename({"DECLARANT_ISO":"PARTNER_ISO","PARTNER_ISO":"DECLARANT_ISO"},axis=1)[columns]]
    df_to_scale = pd.concat(df_eu_l)
    df_to_scale["country_from"] = df_to_scale.PARTNER_ISO
    df_to_scale["country_to"] = df_to_scale.DECLARANT_ISO # cannot look at flow anymore from now on
    return df_to_scale

def extract_table_for_graph(df_in, y="2020", flow="all", criterio="VALUE_IN_EUROS", scale_by=None, pop_df=None, transport_mode=None, threshold_cs=1., threshold_abs=0.):

    df_to_filter = df_in.copy()
    filters = True
    if "TRANSPORT_MODE" in df_to_filter.columns:
        if transport_mode is None:
            df_to_filter = df_to_filter.groupby(df_to_filter.columns[:-2].to_list(),as_index=False)[criterio].sum()
        else:
            filters = filters & (df_to_filter.TRANSPORT_MODE == transport_mode)
    
    ## COUNTRIES WITHOUT POPULATION OR AGGREGATES
    for c in set(df_to_filter.DECLARANT_ISO).union(set(df_to_filter.PARTNER_ISO)):
        if c not in pop_df.index:
            filters &= (df_to_filter.DECLARANT_ISO != c) & (df_to_filter.PARTNER_ISO != c)

    if flow != "all":
        filters &= (df_to_filter.FLOW == flow)
        
    if isinstance(filters,bool):
        df_to_combine = df_to_filter.reset_index(drop=True)
    else:
        df_to_combine = df_to_filter[filters].reset_index(drop=True)

    if flow == "all":
        df_to_scale = combine_flows(df_to_combine, df_in.columns)
    else:
        if flow == 1:
            country_from = "PARTNER_ISO"
            country_to = "DECLARANT_ISO"
        elif flow == 2:
            country_from = "DECLARANT_ISO"
            country_to = "PARTNER_ISO"

    assert scale_by in [None, 'population','in_edges','out_edges']
    if scale_by == 'in_edges':
        # DEPRECATED
        divider = df_to_scale.groupby("country_to",as_index=False)[criterio].sum()
        df = df_to_scale.merge(divider,on="country_to",suffixes=("","_TOT"))
        df[criterio+"_SCALED"] = df[criterio] / df[criterio+"_TOT"]
        df = df[df[criterio+"_SCALED"] >= threshold_abs].sort_values(criterio+"_SCALED",ascending=False)
    elif scale_by == 'population':
        if pop_df is None:
            pop_df, eu_iso = load_population_df()
        # y_ = "2020" if y == "2021" else y
        df_out = df_to_scale.merge(pop_df[[y]],left_on="country_to",right_on="iso2")
        for crit in ["VALUE_IN_EUROS","QUANTITY_IN_KG"]:
            df_out[crit+"_SCALED"] = df_out[crit] / df_out[y]
            df_out[crit+"_RESCALED"] = df_out[crit+"_SCALED"] / df_out[crit+"_SCALED"].sum()
            # df["VALUE_IN_EUROS_MM"] = MinMaxScaler().fit_transform(df["VALUE_IN_EUROS_SCALED"].to_numpy().reshape((-1,1)))[:,0]
        df_out.sort_values("VALUE_IN_EUROS_RESCALED",ascending=False,inplace=True)
        df_out = df_out[df_out["VALUE_IN_EUROS_RESCALED"].cumsum() <= threshold_cs].reset_index(drop=True)
        # Old method
        # df = df[df["VALUE_IN_EUROS_SCALED"] > threshold].sort_values(criterio+"_SCALED",ascending=False)
    elif scale_by == 'out_edges':
        df_out = df_to_scale.copy()
    else:
        df_out = df_to_scale.copy()
    return df_out.rename_axis(y,axis=1)#, df_to_scale

def get_world_countries(df_pop, eu_iso, year=2020):
    y = "2020" if str(year) == "2021" else str(year) 
    df_pop_dict = df_pop[str(y)].to_dict()
    nodes = []
    eu_list = eu_iso.tolist()
    for k in df_pop_dict:
        nodes.append((k,{"pop":df_pop_dict[k],"eu":(k in eu_list)}))
    return nodes

def calc_metrics(G,y="2021"): 

    def vulnerability(in_deg):
        """Opposite of in-degree-percentage: (1-d)%
        """
        vulner = {}
        for k, v in in_deg.items():
            if v != 0 :      
                vulner[k] = 1 - v
            else:
                vulner[k] = 0 
        return vulner

    def adjusted_density(G,y):
        n_ctr = G.number_of_nodes() # which is len(df_pop.iso2.unique())
        n_ctr_eu = 27 if int(y) >= 2020 else 28
        n_ctr_ex = n_ctr - n_ctr_eu
        return G.number_of_edges() / (n_ctr_eu*(n_ctr_eu-1)+2*n_ctr_ex*n_ctr_eu)

    metrics={
             "edges": G.number_of_edges(),
             "nodes": G.number_of_nodes(),
             "size": G.size("weight"),
             "degree": dict(nx.degree(G)),
             "degree_perc": nx.degree_centrality(G),
             "out_degree": dict(G.out_degree),
             "out_degree_perc": nx.out_degree_centrality(G),
             "in_degree": dict(G.in_degree),
             "in_degree_perc": nx.in_degree_centrality(G),
             "density": nx.density(G),
             "density_adj": adjusted_density(G,y),
             "clustering": nx.clustering(G),
             "page_rank": nx.pagerank(G)
            # "betweenness_centrality":nx.betweenness_centrality(G), # SP not interesting
            # "hubness": nx.closeness_centrality(G.to_undirected()) # SP not interesting
        }
    
    metrics["vulnerability"] = vulnerability(metrics["in_degree_perc"])

    if nx.number_of_nodes(G) != 0:
        adj_mat = nx.adjacency_matrix(G).toarray()
        adj_df = pd.DataFrame(adj_mat,columns=G.nodes(),index=G.nodes())
        metrics["in_weight_degree"] = adj_df.sum(axis=0).to_dict()
        metrics["out_weight_degree"] = adj_df.sum(axis=1).to_dict()

    for degree in ["degree","in_degree","out_degree","in_weight_degree","out_weight_degree"]:
        # if len(np.unique(metrics[degree])) <= 2:
        #     metrics[degree+"_gamma"] = np.nan
        #     metrics[degree+"_xmin"] = 0
        # else:
            # try:
            # d_s = pd.Series(metrics[degree])
            # x, y = d_s[d_s != 0].value_counts().index, d_s[d_s != 0].value_counts().values
            # y = y / y.sum()
            # y_log, x_log = np.log(y), np.log(x)
            # power_law = lambda x, a, b: a*(x**b)
            # par, cov = curve_fit(f=power_law,xdata=x,ydata=y)
            # metrics[degree+"_exponent"] = par[1]
            # metrics[degree+"_coef"] = par[0]
        results = powerlaw.Fit([metrics[degree][k] for k in metrics[degree]],verbose=False,xmin=1.)
        metrics[degree+"_gamma"] = results.power_law.alpha
        metrics[degree+"_xmin"] = results.power_law.xmin
    
    return pd.DataFrame(metrics)

def makeGraph(tab_edges, tab_nodes=None, pos_ini=None, weight_flag=False, criterio="VALUE_IN_EUROS", compute_metrics=True, compute_layout=False, lay_dist=5):
    
    G = nx.DiGraph()
    if tab_nodes is not None:
        # df_pop, eu_iso = load_population_df()
        # tab_nodes = get_world_countries(df_pop, eu_iso, year=int(tab_edges.columns.name))
        G.add_nodes_from(tab_nodes)

    edges = []
    if weight_flag == True:
        for i,j,w in tab_edges.loc[:,["country_from","country_to",criterio+"_SCALED"]].values:
            if i == "EU" or j == "EU":
                print(i,j)
            edges += [(i,j,w)]
    else:
        for i,j in tab_edges.loc[:,["country_from","country_to"]].values:
            if i == "EU" or j == "EU":
                print(i,j)
            edges += [(i,j,1)]

    G.add_weighted_edges_from(edges)

    MetricG = None
    coord = None

    # Calcolo le metriche
    if compute_metrics:
        MetricG = calc_metrics(G, y=tab_edges.columns.name)
        MetricG = MetricG.merge(tab_edges.groupby(["country_from"])[criterio].sum().rename("out_weight_abs"),left_index=True,right_index=True,how="left")\
                        .merge(tab_edges.groupby(["country_to"])[criterio].sum().rename("in_weight_abs"),left_index=True,right_index=True,how="left").fillna(0)
        MetricG.index.name = "country"
    
    if compute_layout:
        if pos_ini is None:
            pos_ini = {}
            random.seed(8)
            for node in G.nodes():
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                pos_ini[node] = np.array([x,y])
        coord = nx.spring_layout(G,weight=None,k=lay_dist/math.sqrt(G.order()),iterations=1000)

    return coord, MetricG, G
    

    
def load_population_df(y_from=2000,y_to=2021):
    if not os.path.exists("./data-samples/population/df_pop.csv"):
        ## OLD VERISON
        # pop_un = pd.read_csv("./data-samples/population/population_un.csv",na_values="",keep_default_na=False)
        # code_to_iso3 = pd.read_csv("./data-samples/population/countryUN_to_iso3.csv",na_values="",keep_default_na=False)
        # iso3_to_iso2 = pd.read_csv("./data-samples/population/iso3_to_iso2.csv",na_values="",keep_default_na=False)
        # df_pop_merge = pop_un.merge(code_to_iso3[['LocID', 'ISO3_Code']],left_on='LocID',right_on='LocID').merge(iso3_to_iso2[['alpha-3','alpha-2']], left_on='ISO3_Code',right_on='alpha-3')
        # df_pop = df_pop_merge[df_pop_merge.columns[-1:].append(df_pop_merge.columns[7:-4])].astype({str(y):float for y in range(1950,2021)})
        # df_pop_clean = df_pop.rename({"alpha-2":"iso2"},axis=1)[["iso2"]+[str(y) for y in range(y_from,y_to)]].sort_values("iso2").reset_index(drop=True).fillna("NA") #, df_pop_merge
        # df_pop_clean.to_csv("./data-samples/population/df_pop.csv",index=False)
        # return df_pop_clean.set_index("iso2"), pd.read_table("./data-samples/population/eu_iso2.txt")["Alpha-2"]

        ## NEW VERSION
        un_df = pd.read_csv("./data-samples/population/WPP2022_TotalPopulationBySex.csv",na_filter=False)
        un_df_pivot = un_df[(un_df.ISO2_code != "")&(un_df.Time <= 2022)&(un_df.Variant == "Medium")]\
                            .set_index(['LocID', 'ISO3_code', 'ISO2_code', 'SDMX_code','Location'])\
                            [["Time","PopTotal"]].pivot(columns=["Time"]).droplevel(-2,axis=1).reset_index()
        un_df_pivot.rename({"ISO3_code":"iso3","Location":"Country","ISO2_code":"iso2"},axis=1).to_csv("./data-samples/population/df_pop.csv")
        return un_df_pivot.set_index("iso2"), pd.read_table("./data-samples/population/eu_iso2.txt")["Alpha-2"]
    else:
        return pd.read_csv("./data-samples/population/df_pop.csv",na_filter=False).set_index("iso2"), pd.read_table("./data-samples/population/eu_iso2.txt")["Alpha-2"]

def get_cat_name(table,code,files_dict=files_dict):
    assert table in ["full","tr","tr_intra_"]
    nom = files_dict[table]
    try:
        name = nom[nom["code"]==code].iloc[0]["name"]
        return name
    except:
        # print("Index not existing")
        return "Unknown"

def node_metrics_ts(metrics_df, node, prod=None):
    if prod is None:
        prod = "TO"
    if node != "world":
        return metrics_df[(metrics_df.country == node)&(metrics_df["prod"]==prod)].reset_index(drop=True).copy()
    else:
        fmdf_gr = metrics_df.groupby(["year","month","prod"],as_index=False).sum()
        return fmdf_gr[fmdf_gr["prod"] == prod].reset_index(drop=True).copy()

# def node_metrics_ts(metrics_dict, node, prod=None):
#     node_rows = []
#     for k in metrics_dict:
#         ym, _, p = k.split("_")
#         if prod == None:
#             prod = p
#         if prod == p:
#             node_rows.append(pd.concat([pd.Series(ym,index=["date"],name="IT"),metrics_dict[k].loc["IT"]]))
#     return pd.DataFrame(node_rows).set_index("date")
#     # return node_rows

def plot_bar_metr(metrics_df,country="IT",metr="density",log=False):
    ita_in = metrics_df[(metrics_df.country == country)&(metrics_df["prod"] != "TO")]
    # ita_in = metrics_df[(metrics_df.country == "IT")]
    ita_pivot = ita_in[["year","month","prod",metr]].reset_index(drop=True).pivot(index=["year","month"], columns=["prod"])
    if not log:
        ita_pivot.mean(axis=0).reset_index(level=0,drop=True).rename({i:i+" "+get_cat_name("full",i)[:30] for _,i in ita_pivot.columns},axis=0)\
                 .sort_values(ascending=False).iloc[:25].plot.bar(figsize=(20,8))
        plt.ylabel(metr)
    else:
        np.log10(ita_pivot.mean(axis=0)).reset_index(level=0,drop=True).rename({i:i+" "+get_cat_name("full",i)[:30] for _,i in ita_pivot.columns},axis=0)\
                 .sort_values(ascending=False).iloc[:25].plot.bar(figsize=(20,8))
        plt.ylabel("log "+metr)
    # plt.xticks(rotation=45, rotation_mode="anchor", position=(0.1,-0.11))
    plt.axhline(y=0,color="k",linewidth=0.5)
    if country in ["density","density_adj"]:
        plt.title("Mean "+ metr +" across time for "+country)
    else:
        plt.title("Mean "+ metr +" across time")
    # plt.show()
    
def plot_cat_ts(metrics_df, prod=None, metr="density", country="world", moving_avg=None, log=False):
    df = node_metrics_ts(metrics_df,node=country,prod=prod)
    df["date"] = df.year + df.month
    df.set_index("date",inplace=True)
    title = metr + " of " + country + " for " + get_cat_name("full",prod) + " from 2001 to 2021"
    if moving_avg is None:
        if log:
            np.log10(df[[metr]]).plot(figsize=(20,8),label=metr)
            title = "Log " + title 
        else:
            df[[metr]].plot(figsize=(20,8),label=metr)
    else:
        if log:
            df['MA'] = np.log10(df[metr]).rolling(window=moving_avg).mean()
            title = "Log " + title + ", with rolling average of " + str(moving_avg) + " months"
        else:
            df['MA'] = df[metr].rolling(window=moving_avg).mean()
            title += ", with rolling average of " + str(moving_avg) + " months"
        df.MA.plot(figsize=(20,8),label=metr)
    plt.legend()
    plt.ylabel(metr)
    plt.title(title,fontsize=14)
    # plt.grid()


def plot_power_law(metr_df, metric, discrete=False, pw_plot=False, xmin=0,bins=100):
    y, x = np.histogram(metr_df[metric],bins=bins,density=True)
    x = np.array([np.mean([x[i],x[i+1]]) for i in range(len(x)-1)])
    x = x[y > 0]
    y = y[y > 0]
    x_log = np.log10(x)
    y_log = np.log10(y)
    results = powerlaw.Fit(metr_df[metric].to_numpy(),xmin=xmin,discrete=discrete)
    print("Power law degree is:", results.power_law.alpha)
    print("Power law xmin is:", results.power_law.xmin)
    if pw_plot:
        results.plot_pdf(original_data=True,linear_bins=False)
    else:
        ## Graph
        l = np.linspace(np.min(x),np.max(x))
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))
        # ax1
        ax1.plot(l,l**(-results.power_law.alpha),color="r",label=f"powerlaw, gamma = {results.power_law.alpha:.4}")
        ax1.scatter(x,y,label=metric+" distribution")
        ax1.set_xlabel(metric)
        ax1.set_ylabel("P("+metric+")")
        ax1.legend()
        # ax2
        l = np.linspace(np.min(x_log),np.max(x_log))
        ax2.plot(l,l*(-results.power_law.alpha),color="r",label=f"powerlaw, gamma = {results.power_law.alpha:.4}")
        ax2.scatter(x_log,y_log,label=metric+" distribution")
        ax2.set_xlabel("log("+metric+")")
        ax2.set_ylabel("log( P("+metric+") )")
        ax2.legend()

        plt.suptitle("Power Law fit of "+metric,fontsize=16)
    plt.show()

print("Functions loaded!")