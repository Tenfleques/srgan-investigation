#!/bin/python
import os 
import argparse

parser = argparse.ArgumentParser(description='Log to json creator')
parser.add_argument('-f', '--filename', default='', type=str, help='the source log file')

parser.add_argument('-o', '--out', default='', type=str, help='the destination json filename')

def main():
    args = parser.parse_args()
    json = log_to_json(args.filename, args.out)

    if json:
        out  = args.out
        if not out:
            out = args.filename.replace("log", "json")
        outfile = open(out, "w+")
        outfile.write(json)
        outfile.close()

def log_to_json(filename, output = ""):
    try:
        file_src = open(filename)
        file_contents = file_src.readlines()
        file_src.close()

        json = "["
        for line in file_contents:
            arr = line.split()
            kv = ""

            for i in range(0, len(arr), 2):
                if arr[i] == "time:": # replace s 
                    arr[i + 1] = arr[i + 1].replace("s","")

                arr[i + 1] = arr[i + 1].replace(",", "")

                if not all([c.isdigit() or c == '.' for c in arr[i + 1]]):
                    arr[i + 1] = "\"" + arr[i + 1] + "\""

                arr[i] = "\"" + arr[i].replace(":","\":")

                if kv:
                    kv += ","

                kv += arr[i] + arr[i + 1] 

            if not json == "[":
                json += ","
            
            json += "{" + kv + "}"
        
        return json + "]"   
        
    except OSError as e:
        raise

    


if __name__ == "__main__":
    main()
    