from Bio import Entrez
import json, string, re, os, time
from pathlib import Path
from tqdm import tqdm
import requests
import xml.etree.ElementTree as ET


ENTREZ_EMAIL = "jpvviguini@gmail.com"

if ENTREZ_EMAIL == "your_email@gmail.com":
    print("You have to provide a valid Entrez email.")
    

def list_from_txt(file_path):
    '''
    Creates a list of itens based on a .txt file, each line becomes an item.
    '''

    strings_list = []
    try:
        with open (file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                strings_list.append(line.rstrip('\n'))
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        print("Please verify the path or create this file.")
        return []
    
    return strings_list

# this was limiting each query by 10,000 results per search term.
# def search(query):
#     '''
#     Executes ESearch on pubmed for each item
#     '''

#     final_query = '{} AND English[Language]'.format(query)

#     Entrez.email = ENTREZ_EMAIL
#     handle = Entrez.esearch(db='pubmed', 
#                             sort='relevance', 
#                             retmax='999999', # takes all IDs
#                             retmode='xml', 
#                             term=final_query)
#     results = Entrez.read(handle)
#     handle.close()

#     return results



def search_all(query):
    """
    Executes a full Europe PMC search using its REST API.
    """
    final_query = f'{query} AND LANG:"eng"'
    print(f"\nSearching all results for: {query} (using Europe PMC)")
    
    all_ids = []
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    params = {
        "query": final_query,
        "format": "json",
        "pageSize": 1000,
        "cursorMark": "*"
    }
    
    try:
        while True:
            r = requests.get(url, params=params)
            r.raise_for_status() 
            
            data = r.json()
            
            current_page_results = data.get("resultList", {}).get("result", [])
            
            if not current_page_results:
                break 
            
            for result in current_page_results:
                if 'pmid' in result: 
                    all_ids.append(result['pmid'])
            
            next_cursor = data.get("nextCursorMark")
            
            if not next_cursor or next_cursor == params["cursorMark"]:
                break
            
            params["cursorMark"] = next_cursor
            time.sleep(1.0)

        print(f"  -> Retrieved {len(all_ids)} IDs in total using Europe PMC.")
        return list(set(all_ids)) 

    except requests.exceptions.RequestException as e:
        print(f"  -> Error running Europe PMC search: {e}")
        return all_ids 
    except Exception as e:
        print(f"  -> Error during Europe PMC search: {e}")
        return all_ids


def fetch_details(id_list):
    '''
    Executes a search in Europe PMC for a list of PMIDs
    '''
    if not id_list:
        return {}

    all_papers_list = []
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    # EPMC queries have a length limit, so we chunk the ID list
    chunk_size = 50 
    
    for i in range(0, len(id_list), chunk_size):
        chunk_ids = id_list[i:i + chunk_size]
        
        # create a query of OR PMIDs
        id_query = " OR ".join([f"EXT_ID:{pmid}" for pmid in chunk_ids])
        
        params = {
            "query": id_query,
            "format": "json",
            "pageSize": chunk_size,
            "resultType": "core"  
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            results = data.get("resultList", {}).get("result", [])
            all_papers_list.extend(results)
            
            time.sleep(1.0) 
            
        except requests.exceptions.RequestException as e:
            print(f"  -> Error fetching batch: {e}")
            continue

    # simulate the Entrez dictionary structure to minimize changes in the main loop
    return {'PubmedArticle': all_papers_list}


def flat(lis):
    '''
    aux function to flatten list
    '''
    flatList = []
    for element in lis:
        if type(element) is list:
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList



def extend_gene_search(gene_symbols):
    """
    Search synonyms at NCBI Gene for a list of gene symbols.
    """
    Entrez.email = ENTREZ_EMAIL
    all_gene_aliases = set()
    
    print(f"Search synonyms at NCBI Gene for {len(gene_symbols)} genes...")

    for symbol in gene_symbols:
        if not symbol: continue
        
        print(f"  -> Searching: {symbol}")
        all_gene_aliases.add(symbol.lower()) # adds the symbol itself
        
        try:
            
            # finds the gene's ID at NCBI
            search_term = f'"{symbol}"[Gene Symbol] AND "homo sapiens"[Organism]'
            handle = Entrez.esearch(db="gene", term=search_term, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            if not record['IdList']:
                print(f"    - Symbol {symbol} not found in Homo Sapiens.")
                continue
                
            gene_id = record['IdList'][0]

            
            # search the gene record using the ID
            handle = Entrez.esummary(db="gene", id=gene_id)
            record = Entrez.read(handle)
            handle.close()
            
          
            # extracts symnbol and the official name
            summary = record['DocumentSummarySet']['DocumentSummary'][0]
            
            # adds official name
            if 'Name' in summary:
                all_gene_aliases.add(summary['Name'].lower())
            
            # adds complete official name
            if 'Description' in summary:
                all_gene_aliases.add(summary['Description'].lower())

            # adds other aliases
            if 'OtherAliases' in summary and summary['OtherAliases']:
                aliases = summary['OtherAliases'].split(', ')
                for alias in aliases:
                    all_gene_aliases.add(alias.lower())
           
            time.sleep(0.5) 

        except Exception as e:
            print(f"    - Error when processing {symbol}: {e}")
            time.sleep(1) 

    print(f"Synonyms search complete. Total of unique terms: {len(all_gene_aliases)}")
    return list(all_gene_aliases)


if __name__ == '__main__':
    destination_path = '../results_als_motor_neuron_disease/'
    ids_file_path = '../data/ids_als_motor_neuron_disease.txt'

    Path(ids_file_path).touch(exist_ok=True)
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    
    # loads the general search strings
    search_strings = list_from_txt('../data/search_strings_motor_neuron_disease.txt')
    papers_counter = 0

    
    search_strings = list(dict.fromkeys(search_strings)) 

    print(f"\n--- Starting search in Europe PMC with {len(search_strings)} terms in total ---")

    ids = set()
    try:
        old_papers = list_from_txt(ids_file_path)
        if len(old_papers) > 0:
            ids = set(old_papers)
            print(f"Found {len(ids)} PMIDs already downloaded from previous executions.")
            
    except:
        pass

   # main search loop
    for s in search_strings:
        try:
           
            s_clean_folder_name = s.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
            if not s_clean_folder_name: continue 

           
            s = s.encode('ascii', 'ignore').decode('ascii')
            print(f'Searching for: "{s}"')

            # results = search(s)
            # id_list = results.get('IdList', [])
            # id_list = [x for x in id_list if x not in ids] # Filter IDs already seen

            id_list = search_all(s)
            id_list = [x for x in id_list if x not in ids] # Filter IDs already seen
            
            papers_retrieved = len(id_list)
            if papers_retrieved == 0:
                print('New articles not found.')
            
            else:
                print(f'{papers_retrieved} new articles found.')
                ids.update(id_list)

            batch_size = 9000  
            term_folder = os.path.join(destination_path, s_clean_folder_name)
            Path(term_folder).mkdir(parents=True, exist_ok=True)

            new_papers_in_batch = 0

            for i in tqdm(range(0, len(id_list), batch_size), desc=f"Fetching {s}", unit="batch"):
                
                batch_ids = id_list[i:i + batch_size]
                papers = fetch_details(batch_ids)

                for paper in papers.get('PubmedArticle', []):
                    article_title = ''
                    article_title_filename = ''
                    article_abstract = ''
                    article_year = ''
                    filename = ''
                    path_name = ''

                    try:
                        article_title = paper.get('title', '')
                        if article_title:
                            article_title_filename = article_title.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
                        else:
                            continue 
                    except Exception:
                        continue 

                    if article_title:
                        
                        article_abstract = paper.get('abstractText', '')
                        article_year = paper.get('pubYear', "XXXX")
                        
                        if len(article_year) != 4:
                            article_year = "XXXX"

                        filename = '{}_{}'.format(article_year, article_title_filename)
                        if len(filename) > 150:
                            filename = filename[0:146]

                        path_name = os.path.join(term_folder, f'{filename}.txt')
                        path_name = path_name.encode('ascii', 'ignore').decode('ascii')

                        with open(path_name, "w", encoding='utf-8') as myfile: 
                            myfile.write(article_title + ' ' + article_abstract)
                        
                        new_papers_in_batch += 1

                print(f"  -> Saved {new_papers_in_batch} so far...")
                time.sleep(0.5)  


            
            papers_counter += new_papers_in_batch

            # save new IDs in a log file
            with open(ids_file_path, 'a+', encoding='utf-8') as f:
                for pmid in id_list:
                    f.write('\n' + str(pmid))

        except Exception as e:
            print(f'Fatal error in search loop "{s}": {e}')
            continue


            
    print(f'Data loader finished. Total of {papers_counter} new articles in this section.')