import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Ngram_compute_n_gram_ids_runner(OpBenchmarkBase):
    def __init__(self,name,config):
        super().__init__(name,config)

        self.ne_n=config.get("ne_n")
        self.ne_k=config.get("ne_k")
        self.B=config.get("batch_size")
        self.max_context_len=config.get("max_context_len")

        self.req_lens=config.get("req_lens",[5,4,6])
        self.column_starts_config=config.get("column_starts",[0,2,1])

        self.token_num=sum(self.req_lens)
        self.num_configs=(self.ne_n-1)*self.ne_k

        self.output_shape=(self.token_num,self.num_configs)


    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",self.config.get("dtype",str(self.dtype)))
        state.add_summary("Shape","("+" ".join(map(str,self.output_shape))+")")
        state.add_summary("NE_N",self.ne_n)
        state.add_summary("NE_K",self.ne_k)
        state.add_summary("Batch",self.B)
        state.add_summary("MaxContextLen",self.max_context_len)

        total_elements=self.token_num*self.num_configs

        state.add_element_count(total_elements)

        element_size=4

        weights_elements=(self.ne_n-1)*self.ne_k*self.ne_n
        mods_elements=(self.ne_n-1)*self.ne_k
        embedder_elements=self.num_configs+1
        req_sum_elements=self.B+1
        token_table_elements=self.B*self.max_context_len
        row_indices_elements=self.B
        column_starts_elements=self.B
        output_elements=total_elements

        read_elements=weights_elements+mods_elements+embedder_elements+req_sum_elements+token_table_elements+row_indices_elements+column_starts_elements

        state.add_global_memory_reads(read_elements*element_size)
        state.add_global_memory_writes(output_elements*element_size)


    def _prepare_inputs(self,dev):
        exclusive_req_len_sums=[0]

        for req_len in self.req_lens:
            exclusive_req_len_sums.append(exclusive_req_len_sums[-1]+req_len)

        exclusive_req_len_sums=torch.tensor(exclusive_req_len_sums,dtype=torch.int32,device=dev)

        row_indices=torch.tensor(list(range(self.B)),dtype=torch.int64,device=dev)

        column_starts=torch.tensor(self.column_starts_config,dtype=torch.int32,device=dev)

        ne_token_table=torch.empty((self.B,self.max_context_len),dtype=torch.int32,device=dev)

        for row in range(self.B):
            start=row*self.max_context_len+1
            values=torch.arange(start,start+self.max_context_len,dtype=torch.int32,device=dev)
            ne_token_table[row].copy_(values)

        if self.B>1 and self.max_context_len>4:
            ne_token_table[1,4]=-1

        ne_weights=torch.empty((self.ne_n-1,self.ne_k,self.ne_n),dtype=torch.int32,device=dev)

        for n in range(self.ne_n-1):
            for k in range(self.ne_k):
                for j in range(self.ne_n):
                    ne_weights[n,k,j]=(n+1)*(k+1)+j

        ne_mods=torch.empty((self.ne_n-1,self.ne_k),dtype=torch.int32,device=dev)

        for n in range(self.ne_n-1):
            for k in range(self.ne_k):
                ne_mods[n,k]=97+n*6+k*4

        exclusive_ne_embedder_size_sums=torch.arange(self.num_configs+1,dtype=torch.int32,device=dev)*100

        return ne_weights,ne_mods,exclusive_ne_embedder_size_sums,exclusive_req_len_sums,ne_token_table,row_indices,column_starts


    def prepare_and_get_launcher(self,dev_id,tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            ne_weights,ne_mods,exclusive_ne_embedder_size_sums,exclusive_req_len_sums,ne_token_table,row_indices,column_starts=self._prepare_inputs(dev)
            output=torch.empty(self.output_shape,dtype=torch.int32,device=dev)

        return self.make_launcher(dev_id,torch.ops._C.ngram_compute_n_gram_ids,self.ne_n,self.ne_k,ne_weights,ne_mods,exclusive_ne_embedder_size_sums,exclusive_req_len_sums,ne_token_table,row_indices,column_starts,output)


    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        ne_weights,ne_mods,exclusive_ne_embedder_size_sums,exclusive_req_len_sums,ne_token_table,row_indices,column_starts=self._prepare_inputs(dev)

        output=torch.empty(self.output_shape,dtype=torch.int32,device=dev)

        torch.ops._C.ngram_compute_n_gram_ids(self.ne_n,self.ne_k,ne_weights,ne_mods,exclusive_ne_embedder_size_sums,exclusive_req_len_sums,ne_token_table,row_indices,column_starts,output)

        ref=torch.empty_like(output)

        batch_size=exclusive_req_len_sums.numel()-1
        max_context_len=ne_token_table.shape[1]

        ne_weights_flat=ne_weights.reshape(-1)
        ne_mods_flat=ne_mods.reshape(-1)

        for req_id in range(batch_size):
            req_begin=int(exclusive_req_len_sums[req_id].item())
            req_end=int(exclusive_req_len_sums[req_id+1].item())

            row_idx=int(row_indices[req_id].item())
            column_start=int(column_starts[req_id].item())

            req_token_table_begin=row_idx*max_context_len

            for i in range(req_begin,req_end):
                current_token_offset=i-req_begin
                current_token_table_index=req_token_table_begin+column_start+current_token_offset

                for n in range(self.ne_n-1):
                    for k in range(self.ne_k):
                        ne_weight_base_idx=n*self.ne_k*self.ne_n+k*self.ne_n
                        ne_mod=int(ne_mods_flat[n*self.ne_k+k].item())

                        n_gram_id=0

                        for j in range(n+2):
                            token_index=current_token_table_index-j

                            if token_index<req_token_table_begin:
                                break

                            token_row=token_index//max_context_len
                            token_col=token_index%max_context_len

                            token_value=int(ne_token_table[token_row,token_col].item())

                            if token_value<0:
                                break

                            weight=int(ne_weights_flat[ne_weight_base_idx+j].item())
                            term=token_value*weight
                            n_gram_id+=term%ne_mod

                        n_gram_id%=ne_mod
                        n_gram_id+=int(exclusive_ne_embedder_size_sums[n*self.ne_k+k].item())

                        output_col=n*self.ne_k+k
                        ref[i,output_col]=n_gram_id

        return self.check_diff(output,ref)