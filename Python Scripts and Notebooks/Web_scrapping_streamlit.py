import nest_asyncio
import time
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from google.colab import userdata
import comet_llm
import streamlit as st

nest_asyncio.apply()

MY_OPENAI_KEY = ""
MY_COMET_KEY = ""

comet_llm.init(project="langchain-web-scraping", api_key=MY_COMET_KEY)

html_tags = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p",
    "span",
    "div",
    "ul", "ol", "li",
    "table", "tr", "th", "td",
    "a",
    "b", "strong",
    "i", "em",
    "blockquote", "q", "cite",
    "code", "pre",
    "form", "input", "textarea", "label",
    "dl", "dt", "dd",
    "article",
    "section",
    "nav",
    "aside",
    "header",
    "footer",
    "main",
    "figure", "figcaption",
    "details", "summary",
    "mark",
    "time"
]

def extract_url(url):
    print(url)
    url_loader = AsyncChromiumLoader([url])
    url_docs = url_loader.load()
    bs_transformer = BeautifulSoupTransformer()
    url_transfornm = bs_transformer.transform_documents(
        url_docs, tags_to_extract=["a"]
    )
    llm = ChatOpenAI(openai_api_key=MY_OPENAI_KEY)
    url_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=15000, chunk_overlap=0)
    url_splits = url_splitter.split_documents(url_transfornm)

    url_schema = {
        "properties": {
            "url": {"type": "string"},
        },
        "required": ["url"],
    }
    url_list = []

    if len(url_splits) > 0:
        start_time = time.time()
        extracted_content = create_extraction_chain(schema=url_schema, llm=llm).run(url_splits[0].page_content)
        end_time = time.time()
        comet_llm.log_prompt(
            prompt=str(url_splits[0].page_content),
            metadata={"schema": url_schema},
            output=extracted_content,
            duration=end_time - start_time,
        )
        url_list = [d['url'] for d in extracted_content]

    return url_list

def extract_content(urls):
    contents = []
    for url in urls:
        url_loader = AsyncChromiumLoader([url])
        content_docs = url_loader.load()
        bs_transformer = BeautifulSoupTransformer()
        content_transfornm = bs_transformer.transform_documents(content_docs, tags_to_extract=html_tags)
        content = content_transfornm[0].page_content
        contents.append(content)
    return contents

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f'Document(page_content="{self.page_content},metadata={self.metadata} ")'

def generate_output(resulted_content):
    llm = ChatOpenAI(openai_api_key=MY_OPENAI_KEY)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=15000, chunk_overlap=0)
    splits = splitter.split_documents(resulted_content)
    schema = {
        "properties": {
            "original_id": {"type": "string", "description": "Unique from source"},
            "aug_id": {"type": "string", "description": "Augmented identifier from the context"},
            "country_name": {"type": "string", "description": "Name of the Country"},
            "country_code": {"type": "string", "description": "ISO 3-letter Country Code"},
            "map_coordinates": {
                "type": "object",
                "description": "Geo Point of the region formatted as {'type': 'Point', 'coordinates': [longitude, latitude]}",
                "properties": {"type": {"type": "string"}, "coordinates": {"type": "array", "items": {"type": "number"}}},
            },
            "url": {"type": "string", "description": "Url of the website of the source", "format": "uri"},
            "region_name": {"type": "string", "description": "Region Name for a Country according to World Bank Standards"},
            "region_code": {"type": "string", "description": "Region code for a Region according to World Bank Standards"},
            "Project_title": {"type": "string", "description": "A title for this tender/project used as a headline"},
            "Project_description": {"type": "string", "description": "A summary description of the tender/project"},
            "status": {"type": "string", "description": "The current status of the tender/project from the closed tenderStatus codelist"},
            "stages": {"type": "string", "description": "Stages of the tender/project"},
            "date": {"type": "string", "description": "The date on which the information was first recorded or published", "format": "date"},
            "procurementMethod": {"type": "string", "description": "The procedure used to purchase the relevant works, goods or services"},
            "budget": {"type": "number", "description": "The total upper estimated value of the procurement"},
            "currency": {"type": "string", "description": "The currency for each amount specified using the uppercase 3-letter code from ISO4217"},
            "buyer": {"type": "string", "description": "Entity whose budget will be used to pay for related goods, works or services"},
            "sector": {"type": "string", "description": "A high-level categorization of the main sector this procurement process relates to"},
            "subsector": {"type": "string", "description": "A further subdivision of the sector the procurement process belongs to"},
        },
        "required": [
            "original_id", "aug_id", "country_name", "country_code", "map_coordinates", "url",
            "region_name", "region_code", "title", "description", "status", "stages", "date",
            "procurementMethod", "budget", "currency", "buyer", "sector", "subsector"
        ]
    }

    if len(splits) > 0:
        start_time = time.time()
        extracted_content = create_extraction_chain(schema=schema, llm=llm).run(splits[0].page_content)
        end_time = time.time()

        comet_llm.log_prompt(
            prompt=str(splits[0].page_content),
            metadata={"schema": schema},
            output=extracted_content,
            duration=end_time - start_time,
        )

    return extracted_content     

def main():
    st.title("web scrapper")
    st.write("Enter a URL")
    urls = st.text_area("List Of urls", height=200)
    st.write("Enter your URL List  separated by commas (Ex: url1, url3, url3)")
    urls = [item.strip() for item in urls.split(",")]

    # Submit button
    if st.button("Submit"):
      finalDict=[]
      for url in urls: 
        url_list=extract_url(url)
        url_list.append(url)
        output_content=extract_content(url_list)
        concatenated_content = ' '.join(output_content)
        merged_document = Document(page_content=concatenated_content,metadata="")
        resulted_content = [merged_document]
        final_output=generate_output(resulted_content)

        if(len(final_output)>0):
          finalDict.append(final_output[0])

      df=pd.DataFrame(finalDict)
      st.dataframe(df)

      
if __name__ == "__main__":
    main()      
