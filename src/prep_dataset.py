import xmltodict
from collections import defaultdict
from tqdm import tqdm
path_data = "./data/eudata_unpacked/"

def get_description(description, language_code="en"):
    if description:
        for x in description:
            if x.get("@languageCode")==language_code:
                return x.get("#text")
    return None


labels = defaultdict(lambda: defaultdict())
for path, dirs, files in tqdm(os.walk(path_data)):
    for f in files:
        if os.path.splitext(f)[1]==".xml":
            with open(os.path.join(path, f)) as fd:
                doc = xmltodict.parse(fd.read())
                infos = doc["Transaction"]["TradeMarkTransactionBody"]["TransactionContentDetails"]["TransactionData"]["TradeMarkDetails"]["TradeMark"]
            if infos.get("MarkFeature")=="Figurative":
                nr = infos.get("ApplicationNumber") # application number
                if nr in labels: print("WARINING DUPLICATES")

                labels[nr]["application_date"] = infos.get("ApplicationDate")

                mark_description = infos.get("MarkDescriptionDetails", {}).get("MarkDescription")


                labels[nr]["mark_description"] = get_description([mark_description]) if mark_description and type(mark_description)!=list else get_description(mark_description)
                labels[nr]["displayed_text"] = infos.get("WordMarkSpecification", {}).get("MarkVerbalElementText")

                img_path = infos.get("MarkImageDetails", {}).get("MarkImage", {}).get("MarkImageURI")
                labels[nr]["img_path"] = img_path.split("//")[1] if img_path else None

                labels[nr]["vienna_classification"] = infos.get("MarkImageDetails", {}).get("MarkImage", {}).get("MarkImageCategory", {}).get("CategoryCodeDetails", {}).get("CategoryCode")
                labels[nr]["nice_version"] = infos.get("GoodsServicesDetails", {}).get("GoodsServices", {}).get("ClassificationVersion")



                nice_infos = infos.get("GoodsServicesDetails", {}).get("GoodsServices", {}).get("ClassDescriptionDetails", {}).get("ClassDescription", {})
                if nice_infos and type(nice_infos)!=list: nice_infos = [nice_infos]
                labels[nr]["nice_classification"] = {x.get("ClassNumber"): (get_description([x.get("GoodsServicesDescription")])
                                                                            if type(x.get("GoodsServicesDescription"))!=list and x.get("GoodsServicesDescription")
                                                                            else get_description(x.get("GoodsServicesDescription")) )
                                                     for x in nice_infos}
