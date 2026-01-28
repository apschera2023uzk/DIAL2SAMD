#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Author: Alexander Pschera (apscher1@uni-koeln.de / alexander.pschera@posteo.de)
# This code converts Vaisala DIAL files to the SAMD standard by UHH for daily humr data.

##############################################################

from __future__ import annotations
import glob
import xarray as xr
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, List, Dict
from collections import OrderedDict
from datetime import datetime, date
from typing import Any, Mapping, Optional
import xml.etree.ElementTree as ET

##############################################################################
# 1.5: Parameter
##############################################################################


def _parse_value(raw: str) -> Any:
    s = raw.strip()

    # Leerer Wert
    if s == "":
        return ""

    # String in Anführungszeichen
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # Liste (kommagetrennt)
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [_parse_value(p) for p in parts]

    # Booleans
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False

    # None / Null
    if sl in ("none", "null"):
        return None

    # Integer versuchen
    try:
        return int(s)
    except ValueError:
        pass

    # Float versuchen
    try:
        return float(s)
    except ValueError:
        pass

    # Fallback: String
    return s

##############################################################################

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    config: Dict[str, Any] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Leere Zeilen oder Kommentare überspringen
            if not line or line.startswith("#"):
                continue

            # Inline-Kommentare abschneiden (einfach: ab erstem '#')
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            # key = value erwarten
            if "=" not in line:
                # Wenn du magst, hier warnen oder Fehler werfen
                # print(f"Warnung: Ungültige Zeile in Config: {line}")
                continue

            key, raw_val = line.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()

            config[key] = _parse_value(raw_val)

    return config

##############################################################################

config = load_config("config.txt")
# print(config)

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("~/atris/dial"),
        help="Directory with dial data to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/atris/dial"),
        help="Tagret directory for output file"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=os.path.expanduser("20240805"),
        help="Day to process"
    )    
    return parser.parse_args()

##############################################################################

def read_inputs_of_day(args, file_pattern=config["file_pattern"]):
    root = args.input
    date = args.date
    path = root+"/"+date[0:4]+"/"+date[4:6]+"/"+date[6:8]+"/"    
    
    # files_abs = glob.glob(path+"*ABS*"+date+"*.nc")
    files_wv = glob.glob(path+file_pattern+date+"*.nc")
 
    ds = xr.open_mfdataset(files_wv)
    # 18
    # 1369
    # 20 min => 3*24 * 18 Zeitschirtte: 1296 Zeitschritte...
    
    # print(ds)
    
    return ds
    
##############################################################################

def new_file_name(args, config=config):
    root = args.input
    date = args.date
    path = root+"/"+date[0:4]+"/"+date[4:6]+"/"+date[6:8]+"/"  
    old_file = glob.glob(path+config["kkk"]+"_"+config["sss"]+"_"+config["instr"]+"_"+config["lll"]+"_"+config["var"]+"_*_"+args.date+"000000.nc")
    if old_file==[]:
        vn = 0
    else:
        vn = int((old_file[0].split("_")[-2]).split("v")[-1])+1
    version = f"v{vn:02d}"
    name=config["kkk"]+"_"+config["sss"]+"_"+config["instr"]+"_"+config["lll"]+"_"+config["var"]+"_"+version+"_"+args.date+"000000.nc"
    
    return path+name

##############################################################################

def remove_lowercase_duplicates(attrs: dict) -> dict:
    cleaned = dict(attrs)  # shallow copy

    for key in list(cleaned.keys()):
        if key.islower():
            cap_key = key.capitalize()
            if cap_key in cleaned:
                del cleaned[key]

    return cleaned

############################################################################## 

def order_attrs(ds, desired_order=None, inplace=False):
    """
    Return a dataset with its .attrs ordered.
    """
    if desired_order is None:
        desired_order = ["Title", "Institution", "Contact_person", "Source",\
        "History", "Dependencies", "Conventions", "Processing_date", "Author",\
         "Comments", "License"]

    target = ds if inplace else ds.copy()

    ordered = {k: target.attrs[k] for k in desired_order if k in target.attrs}
    for k, v in target.attrs.items():
        if k not in ordered:
            ordered[k] = v

    target.attrs = ordered
    return target


############################################################################## 


def adhere2metadata_name_convs(ds):
    rename_map = {"title": "Title", "institution": "Institution",\
         "source": "Source", "comment": "Comments",\
         "dependencies": "Dependencies","comments": "Comments",\
         "conventions": "Conventions","conventions": "Conventions",\
         "author": "Author", "license": "License", "authors": "Author",\
         "history": "History"}

    for old, new in rename_map.items():
        if old in ds.attrs:
            ds.attrs[new] = ds.attrs.pop(old)

    return ds    

############################################################################## 

def add_default_metadata(ds):
    """
    Add default global metadata attributes to ds.attrs.
    Hard-coded defaults; can be overwritten elsewhere in the code.
    """
    processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    defaults = {
        "Title": "Vaisala DIAL Atmospheric Profiler DA10-4",
        "Institution":
        "Institute of Geophysics and Meteorology (IGMK) University of Cologne",
        "Source": "DA10-4",
        "History":\
   "Processed with dial2samd.py: https://github.com/apschera2023uzk/DIAL2SAMD",
        "Conventions": "CF-1.8",
        "Comments": "",
        "Dependencies": "external",
        "Processing_date": processing_date,
        "Contact_person": "Ulrich Loehnert (Loehnert@meteo.uni-koeln.de)",
        "Author": "Ulrich Loehnert (Loehnert@meteo.uni-koeln.de)",
        "License": (
            "CC BY 4.0; For non-commercial use only. This data is subject to the "
            "HD(CP)² data policy to be found at https://www.hdcp2.eu and in the "
            "HD(CP)² Observation Data Product standard."
        ),
    }

    # Add defaults without overwriting existing attrs
    for key, value in defaults.items():
        ds.attrs.setdefault(key, value)

    return ds


############################################################################## 

def add_config_values2metadata(ds, config=config):
    for key, var in config.items():
        ds.attrs[key] = var
    return ds

############################################################################## 

def add_samd_attributes(ds: xr.Dataset, config=config) -> xr.Dataset:

    # 1st Rename different naming conventions in ds:
    ds_new = adhere2metadata_name_convs(ds)  

    # 2nd Add all default values to ds (if they are missing):
    ds_new = add_default_metadata(ds_new)

    # 3rd replace dataset or default values by config-values:
    ds_new = add_config_values2metadata(ds_new)

    # 4th order compulsory attributes up:
    ds_new = order_attrs(ds_new)

    return ds_new

############################################################################## 
############################################################################## 
############################################################################## 

def write_samd_metadata_xml(
    ds: xr.Dataset,
    args, outfile,
    config=config,
) -> Path:
    """
    Write a SAMD-style metadata XML file from an xarray Dataset.

    Priority for each field:
      1) config value (if provided and non-empty)  -> overwrite
      2) ds attributes/coords/vars (if present)    -> use
      3) default                                  -> fill

    The XML includes:
      - Table 13 elements (project, region, dsName, fileAverageSize, datatype, provenance,
        contactPersons, keywordLists, temporalExtent, resolution, location, productDescription,
        limitations, references, instruments)
      - Table 14 elements extracted from NetCDF header:
        subtitle (Title), dsAuthor (Author), institution (Institution), conventions (Conventions),
        dependencies (Dependencies), versionHistory (History), license (License), comments (Comments),
        dimensions, variables (name, dimension, standard_name, long_name, units)

    Returns
    -------
    Path to written XML file.
    """
    root = args.input
    date_var = args.date
    path = root+"/"+date_var[0:4]+"/"+date_var[4:6]+"/"+date_var[6:8]+"/"     
    xml_path = path+(outfile.split("/")[-1]).split(".")[0]+".xml"
    xml_path = Path(xml_path)
    print("xml_path: ", xml_path)

    # -----------------------
    # helpers: resolve values
    # -----------------------
    def _is_provided(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() != ""
        return True

    def _get_config(*keys: str) -> Optional[Any]:
        for k in keys:
            if k in config and _is_provided(config[k]):
                return config[k]
        return None

    def _get_ds_attr(*keys: str) -> Optional[Any]:
        for k in keys:
            if k in ds.attrs and _is_provided(ds.attrs[k]):
                return ds.attrs[k]
        return None

    def _resolve(
        config_keys: Tuple[str, ...],
        ds_keys: Tuple[str, ...],
        default: Any,
        *,
        to_str: bool = True,
    ) -> Any:
        v = _get_config(*config_keys)
        if v is None:
            v = _get_ds_attr(*ds_keys)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            v = default
        if to_str:
            return str(v).strip()
        return v

    def _set_text(parent: ET.Element, tag: str, text: Any) -> ET.Element:
        el = ET.SubElement(parent, tag)
        el.text = "" if text is None else str(text)
        return el

    def _safe_date_str(d: Any) -> str:
        # Accept: "YYYY-MM-DD", numpy datetime64, datetime, date
        if d is None:
            return ""
        if isinstance(d, np.datetime64):
            try:
                return str(np.datetime_as_string(d, unit="D"))
            except Exception:
                return str(d)[:10]
        if isinstance(d, datetime):
            return d.strftime("%Y-%m-%d")
        if isinstance(d, date):
            return d.strftime("%Y-%m-%d")
        s = str(d).strip()
        # try to keep first 10 chars for typical ISO
        return s[:10] if len(s) >= 10 else s

    def _infer_temporal_extent_from_ds() -> Tuple[str, str]:
        # Prefer coordinate "time" if present
        if "time" in ds.coords:
            t = ds["time"].values
            if np.size(t) > 0:
                try:
                    tmin = np.nanmin(t)
                    tmax = np.nanmax(t)
                    return _safe_date_str(tmin), _safe_date_str(tmax)
                except Exception:
                    pass
        # fallback: attrs if available
        start = _get_ds_attr("startDate", "StartDate", "time_start")
        end = _get_ds_attr("endDate", "EndDate", "time_end")
        return _safe_date_str(start), _safe_date_str(end)

    def _parse_name_email(s: str) -> Tuple[str, str, str]:
        """
        Parse 'Forename Surname (email)' or 'Forename Surname, email' etc.
        Returns: (forename, surname, email).
        If parsing fails: put full string into surname.
        """
        s = (s or "").strip()
        email = ""
        if "(" in s and ")" in s:
            inside = s[s.find("(") + 1 : s.rfind(")")].strip()
            if "@" in inside:
                email = inside
            s = s[: s.find("(")].strip()

        if "," in s and "@" in s:
            # e.g. "Annika ..., annika@..."
            parts = [p.strip() for p in s.split(",", 1)]
            s = parts[0]
            if len(parts) > 1 and "@" in parts[1]:
                email = parts[1]

        parts = s.split()
        if len(parts) >= 2:
            return parts[0], " ".join(parts[1:]), email
        if len(parts) == 1 and parts[0]:
            return "", parts[0], email
        return "", "", email

    def _dims_for_var(var: xr.DataArray) -> str:
        return ", ".join(list(var.dims)) if hasattr(var, "dims") else ""

    # -----------------------
    # sensible defaults (vitII / DIAL)
    # -----------------------
    defaults = {
        "project": "VITAL II (vitII)",
        "region": "Europe/Germany",
        "dsName": "vitII_dial_l1_any",  # you can override in config
        "fileAverageSize_value": "0",
        "fileAverageSize_unit": "MB",
        "fileAverageSize_status": "unknown",
        "datatype": "daily",
        "provenance": "Compiled DIAL files and harmonized metadata for the VITAL II campaign.",
        "resolution_temporal_value": "60",
        "resolution_temporal_unit": "s",
        "resolution_horizontal_value": "0",
        "resolution_horizontal_unit": "m",
        "resolution_vertical_value": "1",
        "resolution_vertical_unit": "m",
        "location": "Ground-based DIAL instrument at campaign site. Single instrument deployment.",
        "productDescription": "Time-height profiles from a Vaisala DIAL Atmospheric Profiler.",
        "limitations": "none",
        "publicationDate": datetime.now().strftime("%Y-%m-%d"),
        "references_default": "none",
        "instrumentSpecification": "none",
    }

    # -----------------------
    # NetCDF header mapped fields (Table 14)
    # -----------------------
    subtitle = _resolve(("Title", "title", "subtitle"), ("Title", "title"), default="Vaisala DIAL Atmospheric Profiler data")
    institution = _resolve(("Institution", "institution"), ("Institution", "institution"), default="University of Cologne / IGMK")
    conventions = _resolve(("Conventions", "conventions"), ("Conventions", "conventions"), default="CF-1.8")
    dependencies = _resolve(("Dependencies", "dependencies"), ("Dependencies", "dependencies"), default="external")
    version_history = _resolve(("History", "history"), ("History", "history"), default="none")
    license_ = _resolve(("License", "license"), ("License", "license"), default="none")
    comments = _resolve(("Comments", "comments", "comment"), ("Comments", "comments", "comment"), default="none")

    author_str = _resolve(("Author", "author"), ("Author", "author"), default="none")
    author_forename, author_surname, author_email = _parse_name_email(author_str)

    # -----------------------
    # Table 13 fields (top-level)
    # -----------------------
    project = _resolve(("project",), ("project",), default=defaults["project"])
    region = _resolve(("region",), ("region",), default=defaults["region"])
    dsName = _resolve(("dsName", "datasetName"), ("dsName", "datasetName"), default=defaults["dsName"])

    datatype = _resolve(("datatype",), ("datatype",), default=defaults["datatype"])
    provenance = _resolve(("provenance",), ("provenance",), default=defaults["provenance"])
    location = _resolve(("location",), ("location",), default=defaults["location"])
    productDescription = _resolve(("productDescription",), ("productDescription",), default=defaults["productDescription"])
    limitations = _resolve(("limitations",), ("limitations",), default=defaults["limitations"])

    # fileAverageSize
    fas_value = _resolve(("fileAverageSize_value", "fileAverageSize"), ("fileAverageSize",), default=defaults["fileAverageSize_value"])
    fas_unit = _resolve(("fileAverageSize_unit",), ("fileAverageSize_unit",), default=defaults["fileAverageSize_unit"])
    fas_status = _resolve(("fileAverageSize_status",), ("fileAverageSize_status",), default=defaults["fileAverageSize_status"])

    # temporalExtent: config -> ds -> infer
    start_cfg = _get_config("startDate", "temporalExtent_startDate")
    end_cfg = _get_config("endDate", "temporalExtent_endDate")
    if start_cfg is None or end_cfg is None:
        start_ds, end_ds = _infer_temporal_extent_from_ds()
    else:
        start_ds, end_ds = "", ""
    startDate = _safe_date_str(start_cfg) if start_cfg is not None else start_ds
    endDate = _safe_date_str(end_cfg) if end_cfg is not None else end_ds
    if not startDate:
        startDate = defaults["publicationDate"]
    if not endDate:
        endDate = defaults["publicationDate"]

    # resolution
    res_t_val = _resolve(("resolution_temporal_value",), ("resolution_temporal_value",), default=defaults["resolution_temporal_value"])
    res_t_unit = _resolve(("resolution_temporal_unit",), ("resolution_temporal_unit",), default=defaults["resolution_temporal_unit"])
    res_h_val = _resolve(("resolution_horizontal_value",), ("resolution_horizontal_value",), default=defaults["resolution_horizontal_value"])
    res_h_unit = _resolve(("resolution_horizontal_unit",), ("resolution_horizontal_unit",), default=defaults["resolution_horizontal_unit"])
    res_v_val = _resolve(("resolution_vertical_value",), ("resolution_vertical_value",), default=defaults["resolution_vertical_value"])
    res_v_unit = _resolve(("resolution_vertical_unit",), ("resolution_vertical_unit",), default=defaults["resolution_vertical_unit"])

    # contactPersons: allow list in config, else use Contact_person/Contact_person-like
    # config example:
    # config["contactPersons"] = [
    #   {"institution": "...", "forename": "...", "surname":"...", "code":"", "phone":"", "email":"..."},
    # ]
    contact_persons_cfg = _get_config("contactPersons")
    if isinstance(contact_persons_cfg, list) and len(contact_persons_cfg) > 0:
        contact_persons = contact_persons_cfg
    else:
        cp_str = _resolve(("Contact_person", "contact_person"), ("Contact_person", "contact_person"), default="none")
        cp_forename, cp_surname, cp_email = _parse_name_email(cp_str)
        contact_persons = [{
            "institution": institution,
            "forename": cp_forename,
            "surname": cp_surname if cp_surname else cp_str,
            "code": _resolve(("contact_code",), ("contact_code",), default=""),
            "phone": _resolve(("contact_phone",), ("contact_phone",), default=""),
            "email": cp_email,
        }]

    # keywordLists: allow list in config; else fill one default entry
    # config["keywordLists"] = [
    #   {"experimentType":"field campaign", "measurementType":"remote sensing", "mainGroup":"atmosphere",
    #    "variableGroup":"water vapor", "list_number":1}
    # ]
    keyword_lists_cfg = _get_config("keywordLists")
    if isinstance(keyword_lists_cfg, list) and len(keyword_lists_cfg) > 0:
        keyword_lists = keyword_lists_cfg
    else:
        keyword_lists = [{
            "list_number": 1,
            "experimentType": _resolve(("experimentType",), ("experimentType",), default="field campaign"),
            "measurementType": _resolve(("measurementType",), ("measurementType",), default="remote sensing"),
            "mainGroup": _resolve(("mainGroup",), ("mainGroup",), default="atmosphere"),
            "variableGroup": _resolve(("variableGroup",), ("variableGroup",), default="thermodynamic profiles"),
        }]

    # instruments: lat/lon/zsl/z may be in coords or attrs; config overrides.
    def _get_float_from_ds(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in ds.coords:
                try:
                    return float(ds.coords[k].values)
                except Exception:
                    pass
            if k in ds.variables:
                try:
                    da = ds[k]
                    if da.size == 1:
                        return float(da.values.item())
                except Exception:
                    pass
            if k in ds.attrs:
                try:
                    return float(ds.attrs[k])
                except Exception:
                    pass
        return None

    lat = _get_config("lat", "latitude")
    lon = _get_config("lon", "longitude")
    alt = _get_config("zsl", "altitude", "altitude_asl")
    hgt = _get_config("z", "height", "height_agl")

    lat = float(lat) if lat is not None else _get_float_from_ds(["lat", "latitude"])
    lon = float(lon) if lon is not None else _get_float_from_ds(["lon", "longitude"])
    alt = float(alt) if alt is not None else _get_float_from_ds(["zsl", "altitude", "altitude_asl"])
    hgt = float(hgt) if hgt is not None else _get_float_from_ds(["z", "height", "height_agl"])

    # instrument source and spec
    source = _resolve(("Source", "source"), ("Source", "source"), default="Vaisala DIAL Atmospheric Profiler DA10-4")
    instrument_spec = _resolve(("instrumentSpecification",), ("instrumentSpecification",), default=defaults["instrumentSpecification"])

    instruments_cfg = _get_config("instruments")
    if isinstance(instruments_cfg, list) and len(instruments_cfg) > 0:
        instruments = instruments_cfg
    else:
        instruments = [{
            "list_number": 1,
            "source": source,
            "latitude": lat if lat is not None else "",
            "latitude_unit": "degrees_north",
            "longitude": lon if lon is not None else "",
            "longitude_unit": "degrees_east",
            "altitude": alt if alt is not None else "",
            "altitude_unit": "m",
            "height": hgt if hgt is not None else "",
            "height_unit": "m",
            "instrumentSpecification": instrument_spec,
        }]

    # references: optional in standard, but user requested "fülle all diese Werte"
    # config["references"] = [{"referenceType": "...", "publicationDate": "...", "author": "...", ...}]
    references_cfg = _get_config("references")
    if isinstance(references_cfg, list) and len(references_cfg) > 0:
        references = references_cfg
    else:
        references = [{
            "list_number": 1,
            "referenceType": "none",
            "publicationDate": defaults["publicationDate"],
            "author": "none",
            "title": "none",
            "abstract": "none",
            "publish": "none",
        }]

    # -----------------------
    # Build XML
    # -----------------------
    root = ET.Element("samdMetadata")

    # Table 13 (part 1)
    _set_text(root, "project", project)
    _set_text(root, "region", region)
    _set_text(root, "dsName", dsName)

    fileAverageSize_el = ET.SubElement(root, "fileAverageSize")
    _set_text(fileAverageSize_el, "size", fas_value)
    fileAverageSize_el.find("size").set("unit", fas_unit)
    _set_text(fileAverageSize_el, "status", fas_status)

    _set_text(root, "datatype", datatype)
    _set_text(root, "provenance", provenance)

    contactPersons_el = ET.SubElement(root, "contactPersons")
    for i, p in enumerate(contact_persons, start=1):
        p_el = ET.SubElement(contactPersons_el, "list")
        _set_text(p_el, "number", p.get("list_number", i))
        _set_text(p_el, "institution", p.get("institution", institution))
        _set_text(p_el, "forename", p.get("forename", ""))
        _set_text(p_el, "surname", p.get("surname", ""))
        _set_text(p_el, "code", p.get("code", ""))
        _set_text(p_el, "phone", p.get("phone", ""))
        _set_text(p_el, "email", p.get("email", ""))

    keywordLists_el = ET.SubElement(root, "keywordLists")
    for i, kw in enumerate(keyword_lists, start=1):
        kw_el = ET.SubElement(keywordLists_el, "list")
        _set_text(kw_el, "number", kw.get("list_number", i))
        _set_text(kw_el, "experimentType", kw.get("experimentType", ""))
        _set_text(kw_el, "measurementType", kw.get("measurementType", ""))
        _set_text(kw_el, "mainGroup", kw.get("mainGroup", ""))
        _set_text(kw_el, "variableGroup", kw.get("variableGroup", ""))

    temporalExtent_el = ET.SubElement(root, "temporalExtent")
    _set_text(temporalExtent_el, "startDate", startDate)
    _set_text(temporalExtent_el, "endDate", endDate)

    resolution_el = ET.SubElement(root, "resolution")
    t_el = _set_text(resolution_el, "temporal", res_t_val)
    t_el.set("unit", res_t_unit)
    h_el = _set_text(resolution_el, "horizontal", res_h_val)
    h_el.set("unit", res_h_unit)
    v_el = _set_text(resolution_el, "vertical", res_v_val)
    v_el.set("unit", res_v_unit)

    _set_text(root, "location", location)
    _set_text(root, "productDescription", productDescription)
    _set_text(root, "limitations", limitations)

    references_el = ET.SubElement(root, "references")
    for i, r in enumerate(references, start=1):
        r_el = ET.SubElement(references_el, "list")
        _set_text(r_el, "number", r.get("list_number", i))
        _set_text(r_el, "referenceType", r.get("referenceType", "none"))
        _set_text(r_el, "publicationDate", _safe_date_str(r.get("publicationDate", defaults["publicationDate"])))
        _set_text(r_el, "author", r.get("author", "none"))
        _set_text(r_el, "title", r.get("title", "none"))
        _set_text(r_el, "abstract", r.get("abstract", "none"))
        _set_text(r_el, "publish", r.get("publish", "none"))

    instruments_el = ET.SubElement(root, "instruments")
    for i, inst in enumerate(instruments, start=1):
        inst_el = ET.SubElement(instruments_el, "list")
        _set_text(inst_el, "number", inst.get("list_number", i))
        _set_text(inst_el, "source", inst.get("source", source))

        loc_el = ET.SubElement(inst_el, "instrumentLocation")
        lat_el = _set_text(loc_el, "latitude", inst.get("latitude", ""))
        lat_el.set("unit", inst.get("latitude_unit", "degrees_north"))
        lon_el = _set_text(loc_el, "longitude", inst.get("longitude", ""))
        lon_el.set("unit", inst.get("longitude_unit", "degrees_east"))
        alt_el = _set_text(loc_el, "altitude", inst.get("altitude", ""))
        alt_el.set("unit", inst.get("altitude_unit", "m"))
        h_el2 = _set_text(loc_el, "height", inst.get("height", ""))
        h_el2.set("unit", inst.get("height_unit", "m"))

        _set_text(inst_el, "instrumentSpecification", inst.get("instrumentSpecification", instrument_spec))

    # Table 14 (part 2) - extracted from NetCDF header
    netcdf_el = ET.SubElement(root, "netcdfHeader")

    _set_text(netcdf_el, "subtitle", subtitle)

    dsAuthor_el = ET.SubElement(netcdf_el, "dsAuthor")
    _set_text(dsAuthor_el, "forename", author_forename)
    _set_text(dsAuthor_el, "surname", author_surname if author_surname else author_str)
    _set_text(dsAuthor_el, "email", author_email)

    _set_text(netcdf_el, "institution", institution)
    _set_text(netcdf_el, "conventions", conventions)
    _set_text(netcdf_el, "dependencies", dependencies)
    _set_text(netcdf_el, "versionHistory", version_history)
    _set_text(netcdf_el, "license", license_)
    _set_text(netcdf_el, "comments", comments)

    # dimensions
    dims_el = ET.SubElement(netcdf_el, "dimensions")
    for dim_name, dim_len in ds.dims.items():
        d_el = ET.SubElement(dims_el, "dimension")
        _set_text(d_el, "name", dim_name)
        _set_text(d_el, "length", dim_len)

    # variables (include coords + data_vars, like NetCDF header does)
    vars_el = ET.SubElement(netcdf_el, "variables")
    for name, var in ds.variables.items():
        v_el = ET.SubElement(vars_el, "variable")
        _set_text(v_el, "name", name)
        _set_text(v_el, "dimension", _dims_for_var(var))

        # pull common attrs if present
        stdn = var.attrs.get("standard_name", "")
        lonn = var.attrs.get("long_name", "")
        units = var.attrs.get("units", "")

        _set_text(v_el, "standard_name", stdn)
        _set_text(v_el, "long_name", lonn)
        _set_text(v_el, "units", units)

    # -----------------------
    # Write XML with pretty formatting
    # -----------------------
    def _indent(elem: ET.Element, level: int = 0) -> None:
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                _indent(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    _indent(root)

    tree = ET.ElementTree(root)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    return xml_path

##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    
    # Read inputs:
    ds = read_inputs_of_day(args)
    
    # Get new file name:
    outfile = new_file_name(args)
    print("Output filename: ", outfile)

    # Metadata adding from old file, config file and defaults makes sense now:
    # Add global attributes for gb:
    ds = add_samd_attributes(ds)
    
    


    #########
    # These two functions are ugly, because AI wrote them:
    # Write XML file with Metadata:
    write_samd_metadata_xml(ds, args, outfile)
    #########


    


    # Write NetCDF4_CLASSIC file:
    ds.to_netcdf(outfile, format="NETCDF4_CLASSIC")
    
    
    
   

        

