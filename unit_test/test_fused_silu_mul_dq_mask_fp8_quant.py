import torch
import pytest
import mcoplib.op as op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="need CUDA")


class TestFusedSiluMulDqMaskFp8Quant:
    def setup_method(self):
        self.device = "cuda"
        self.default_dtype = torch.bfloat16

    def _make_valid_inputs(self, batch=2, seq_len=128, hidden_dim=64,
                          mask_dtype=torch.int32, input_dtype=torch.bfloat16):
        assert hidden_dim % 2 == 0
        input_tensor = torch.randn(
            batch, seq_len, hidden_dim,
            dtype=input_dtype, device=self.device
        )
        out_tensor = torch.zeros(
            batch, seq_len, hidden_dim // 2,
            dtype=input_dtype, device=self.device
        )
        mask = torch.full(
            (batch,), seq_len,
            dtype=mask_dtype, device=self.device
        )
        return out_tensor, input_tensor, mask

    def test_basic_int32_mask(self):
        out, inp, mask = self._make_valid_inputs(mask_dtype=torch.int32)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 32])
        assert out.device.type == "cuda"

    def test_basic_int64_mask(self):
        out, inp, mask = self._make_valid_inputs(mask_dtype=torch.int64)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 32])

    def test_input_not_contiguous(self):
        out, inp, mask = self._make_valid_inputs()
        inp = inp.transpose(1, 2)
        with pytest.raises(RuntimeError, match="contiguous"):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_output_not_contiguous(self):
        out, inp, mask = self._make_valid_inputs()
        out = out.transpose(1, 2)
        with pytest.raises(RuntimeError, match="contiguous"):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_mask_not_contiguous(self):
      """TORCH_CHECK: mask not contiguous"""
      out, inp, mask = self._make_valid_inputs()

      mask = mask.unsqueeze(0).expand(2, -1)  
      
      assert not mask.is_contiguous()  
    
      with pytest.raises(RuntimeError, match="contiguous"):
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
    
    def test_mask_unsupported_dtype_int16(self):
        out, inp, _ = self._make_valid_inputs()
        mask = torch.full((2,), 128, dtype=torch.int16, device=self.device)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_input_dtype_fp16(self):
        out, inp, mask = self._make_valid_inputs(input_dtype=torch.float16)
        out = torch.zeros(2, 128, 32, dtype=torch.float16, device=self.device)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.dtype == torch.float16

    def test_input_dtype_fp32(self):
        out, inp, mask = self._make_valid_inputs(input_dtype=torch.float32)
        out = torch.zeros(2, 128, 32, dtype=torch.float32, device=self.device)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.dtype == torch.float32
    
    def test_input_dtype_int8(self):
        inp = torch.randint(-128, 127, (2, 128, 64), dtype=torch.int8, device=self.device)
        out = torch.zeros(2, 128, 32, dtype=torch.int8, device=self.device)
        mask = torch.full((2,), 128, dtype=torch.int32, device=self.device)
        with pytest.raises(RuntimeError):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_input_dtype_int64(self):
        inp = torch.randint(-1000, 1000, (2, 128, 64), dtype=torch.int64, device=self.device)
        out = torch.zeros(2, 128, 32, dtype=torch.int64, device=self.device)
        mask = torch.full((2,), 128, dtype=torch.int32, device=self.device)
        with pytest.raises(RuntimeError):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_input_dtype_bool(self):
        inp = torch.randint(0, 2, (2, 128, 64), dtype=torch.bool, device=self.device)
        out = torch.zeros(2, 128, 32, dtype=torch.bool, device=self.device)
        mask = torch.full((2,), 128, dtype=torch.int32, device=self.device)
        with pytest.raises(RuntimeError):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_minimal_shape(self):
        out, inp, mask = self._make_valid_inputs(batch=1, seq_len=1, hidden_dim=2)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([1, 1, 1])

    def test_mask_zero(self):
        out, inp, mask = self._make_valid_inputs()
        mask.fill_(0)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_mask_partial_valid(self):
        out, inp, mask = self._make_valid_inputs(seq_len=128)
        mask[0] = 64
        mask[1] = 100
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 32])
    def test_input_not_contiguous(self):
        """TORCH_CHECK(input.is_contiguous())"""
        out, inp, mask = self._make_valid_inputs()
        inp = inp.transpose(1, 2)  
        assert not inp.is_contiguous()
        with pytest.raises(RuntimeError, match=r"contiguous"):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_output_not_contiguous(self):
        """TORCH_CHECK(out.is_contiguous())"""
        out, inp, mask = self._make_valid_inputs()
        out = out.transpose(1, 2)  
        assert not out.is_contiguous()
        with pytest.raises(RuntimeError, match=r"contiguous"):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)

    def test_mask_not_contiguous(self):
        """TORCH_CHECK(mask.is_contiguous())"""
        out, inp, mask = self._make_valid_inputs()
        mask = mask.unsqueeze(0).expand(2, -1)  
        assert not mask.is_contiguous()
        with pytest.raises(RuntimeError, match=r"contiguous"):
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
            
    def test_hidden_size_minimal(self):
        
        out, inp, mask = self._make_valid_inputs(hidden_dim=2)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 1])
            
    def test_num_tokens_divisible_by_mask_size(self):
       
       
        out, inp, mask = self._make_valid_inputs(batch=2, seq_len=128)
        mask = torch.full((2,), 128, dtype=torch.int32, device=self.device)
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 32])
        assert out.device.type == "cuda"

    def test_out_shape_correct(self):
        
        out, inp, mask = self._make_valid_inputs(hidden_dim=64)
      
        assert out.shape[-1] == 32
        op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
        assert out.shape == torch.Size([2, 128, 32])
        assert out.device.type == "cuda"
        
    def test_hidden_size_various(self):
        
        for h in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
            out = torch.zeros(2, 128, h // 2, dtype=torch.bfloat16, device=self.device)
            inp = torch.randn(2, 128, h, dtype=torch.bfloat16, device=self.device)
            mask = torch.full((2,), 128, dtype=torch.int32, device=self.device)
            op.fused_silu_mul_dq_mask_fp8_quant(out, inp, mask)
            assert out.shape == torch.Size([2, 128, h // 2]), f"hidden_size={h} failed"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])