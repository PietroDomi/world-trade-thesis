import os
import pandas as pd
import numpy as np
import random
import math
from networkx.readwrite import json_graph
import json
import networkx as nx
import pickle
import logging.config
import datetime
DATA_AVAILABLE = "data"+os.sep+"dataAvailable"
SEP = ","
DATA_EXTENTION = ".dat"
NTSR_PROD_FILE = "data"+os.sep+"NSTR.txt"
NTSR_DIGITS = 3 # numero di digits per classificazione Transporti
NODIMAX = 70
INTRA_FILE = "data/cpa_intra/cpa_intra.csv"
EXTRA_FILE = "data/tr_intra_201001.dat"
criterio = "VALUE_IN_EUROS" #VALUE_IN_EUROS 	QUANTITY_IN_KG

logging.config.fileConfig('./logging.conf')
logger = logging.getLogger('graphLog')

def estrai_tabella_per_grafo(tg_period, tg_perc, listaMezzi, flow, product, criterio, selezioneMezziEdges, df_transport_estrazione):
    #estraggo dalla tabella solo le informazioni richieste nei filtri richiesti al runtime
    logging.info("### estrai_tabella_per_grafo...") 
    logging.info("ESTRAGGO TABELLA COMEXT") 
    
    df_transport_estrazione = df_transport_estrazione[df_transport_estrazione["FLOW"]==flow]
    if tg_period is not None:
        tg_period = np.int32(tg_period)
        df_transport_estrazione = df_transport_estrazione[df_transport_estrazione["PERIOD"]==tg_period]
    
    if listaMezzi is not None:    
        df_transport_estrazione = df_transport_estrazione[df_transport_estrazione["TRANSPORT_MODE"].isin(listaMezzi)]
    
    if product is not None:
        df_transport_estrazione = df_transport_estrazione[df_transport_estrazione["PRODUCT"]==product]

    # costruisce una query per eliminare i mezzi in un arco nel grafo
    def build_query_mezzi(selezioneMezziEdges):
        listQuery=[]
        for edge in selezioneMezziEdges:
            From = edge["from"]
            To = edge["to"]
            exclude = str(edge["exclude"])
            listQuery.append("((DECLARANT_ISO == '"+From+"' & PARTNER_ISO == '"+To+"' & TRANSPORT_MODE in "+exclude+")|(DECLARANT_ISO == '"+To+"' & PARTNER_ISO == '"+From+"' & TRANSPORT_MODE in "+exclude+"))")
        return "not ("+("|".join(listQuery))+")"
    
    if selezioneMezziEdges is not None:
        query = build_query_mezzi(selezioneMezziEdges)
        logging.info("QUERY selezione MezziEdge:")
        df_transport_estrazione = df_transport_estrazione.query(query)

    # aggrega indimendentemente dai mezzi o prodotti ed ordina secondo il criterio scelto VALUE o QUANTITY 
    df_transport_estrazione = df_transport_estrazione.groupby(["DECLARANT_ISO","PARTNER_ISO"]).sum().reset_index()[["DECLARANT_ISO","PARTNER_ISO",criterio]]
    df_transport_estrazione = df_transport_estrazione.sort_values(criterio,ascending=False)    
    # taglio sui nodi 
    if tg_perc is not None:
        SUM = df_transport_estrazione[criterio].sum()     
        df_transport_estrazione = df_transport_estrazione[df_transport_estrazione[criterio].cumsum(skipna=False)/SUM*100<tg_perc] 
        logging.info("### estrai_tabella_per_grafo exit")     
    return df_transport_estrazione

def makeGraph(tab4graph, pos_ini, weight_flag, flow, AnalisiFlag): 
    # costruisce sulla base della tabella filtrata
    # il grafo con le relative metriche
    logging.info("### makeGraph... ")     

    def calc_metrics(Grafo, FlagWeight):
        logging.info("### metrics... ")     
        in_deg = nx.in_degree_centrality(Grafo)
        metrics = {}
        vulner = {}
        for k, v in in_deg.items():
            if v!=0:      
                vulner[k] = 1-v
            else:
                vulner[k] = 0            
            metrics={
            "degree_centrality": nx.degree_centrality(Grafo),
            "density": nx.density(Grafo),
            "vulnerability": vulner,
            "exportation strenght": nx.out_degree_centrality(Grafo),
            #
            "hubness": nx.closeness_centrality(Grafo.to_undirected())
            }
        return metrics 

    G = nx.DiGraph()

    # assegno i ruoli IMPORT e EXPORT
    if flow == 1:
        logging.info("FLOW: import")
        country_from = "PARTNER_ISO"
        country_to = "DECLARANT_ISO"
        
    if flow == 2:
        logging.info("FLOW: export")    
        country_from = "DECLARANT_ISO"
        country_to = "PARTNER_ISO"

    # costruisco il grafo con edges e nodi
    # se il grafo è pesato
    # assegno il peso VALUE o QUANTITY in funzione del criterio scelto per ordinare il mercato
    # ed eseguire il taglio      
    if weight_flag == True:
        Wsum = tab4graph[criterio].sum()
        edges = [(i,j,w/Wsum) for i,j,w in tab4graph.loc[:,[country_from,country_to,criterio]].values]
    if weight_flag == False:
        edges = [(i,j,1) for i,j in tab4graph.loc[:,[country_from,country_to]].values]

    G.add_weighted_edges_from(edges)

    # Calcolo le metriche
    MetricG = calc_metrics(G, weight_flag)	
    
    # passo alla rappresentazione json del grafo
    GG = json_graph.node_link_data(G)
    Nodes = GG["nodes"]
    Links = GG["links"]

    if pos_ini is None:
        pos_ini = {}
        random.seed(8)
        for node in Nodes:
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            pos_ini[node['id']] = np.array([x,y])
    else:
            logging.info("-- POSIZIONE DEI NODI PRECEDENTE ACQUISITA --")
    try:
        logging.info(str(pos_ini))
        coord = nx.spring_layout(G,k=5/math.sqrt(G.order()),pos=pos_ini)
        coord = nx.spring_layout(G,k=5/math.sqrt(G.order()),pos=coord) # stable solution
        coord = nx.spring_layout(G,k=5/math.sqrt(G.order()),pos=coord) # stable solution
    except:
        return None,None,None

    #########################################################
    df_coord = pd.DataFrame.from_dict(coord,orient='index')
    df_coord.columns = ['x', 'y']

    df = pd.DataFrame(GG["nodes"])
    df.columns = ['label']
    df['id'] = np.arange(df.shape[0])
    df = df[['id', 'label']]    
    out = pd.merge(df, df_coord, left_on='label', right_index=True)
    dict_nodes = out.T.to_dict().values()
    
    dfe = pd.DataFrame(GG["links"])[["source" , "target"]]
    res = dfe.set_index('source').join(out[['label','id']].set_index('label'), on='source', how='left')
    res.columns = ['target', 'source_id']
    res2 = res.set_index('target').join(out[['label','id']].set_index('label'), on='target', how='left')
    res2.columns = ['from','to']
    res2.reset_index(drop=True, inplace=True)
    dict_edges = res2.T.to_dict().values()

    new_dict = { "nodes": list(dict_nodes), "edges": list(dict_edges), "metriche": MetricG}

    JSON=json.dumps(new_dict) 
    logging.info("### makeGraph exit")   

    return coord, JSON, G