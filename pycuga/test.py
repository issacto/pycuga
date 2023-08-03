import algos.tools as tools
cudaCode = tools.read_files_as_strings("pycuga/algos/cuda")
with open("./test.cu", 'w') as file:
    file.write(cudaCode)
# print(cudaCode)