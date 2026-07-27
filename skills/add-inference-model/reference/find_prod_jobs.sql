-- Phase 5: find real production training jobs to parity-test against.
-- Prod Supabase project: xhcxueykzjjjendgfxyu (table trainer_jobs).
-- Replace <ARCH> with the training arch name WITH the colon, e.g. 'krea2:o_edit'.
-- 'stopped' is the terminal success state for training (NOT 'completed').

-- 1) Inventory: how many jobs per arch, and how many actually have sample prompts.
select
  job_config->'config'->'process'->0->'model'->>'arch'  as arch,
  (job_config->'config'->'process'->0->'model'->>'quantize')     as quantize,
  (job_config->'config'->'process'->0->'model'->>'quantize_te')  as quantize_te,
  count(*) as jobs,
  count(*) filter (
    where jsonb_array_length(
      coalesce(job_config->'config'->'process'->0->'sample'->'samples','[]'::jsonb)) > 0
  ) as with_prompts
from trainer_jobs
where is_deleted is not true
  and status = 'stopped'
  and job_config->'config'->'process'->0->'model'->>'arch' like 'krea%'   -- <-- your family
group by 1,2,3
order by jobs desc;

-- 2) Best parity candidates: quantize=false AND quantize_te=false (clean bf16),
--    with sample prompts, enough steps. SCREEN the prompt text for NSFW before use.
select
  id, name, step, user_id,
  job_config->'config'->'process'->0->>'trigger_word'                 as trigger_word,
  job_config->'config'->'process'->0->'sample'->>'sample_steps'       as steps,
  job_config->'config'->'process'->0->'sample'->>'guidance_scale'     as cfg,
  job_config->'config'->'process'->0->'sample'->>'seed'               as seed,
  job_config->'config'->'process'->0->'sample'->>'walk_seed'          as walk_seed,
  jsonb_array_length(job_config->'config'->'process'->0->'sample'->'samples') as n_prompts,
  left(string_agg(pr->>'prompt', ' || ' order by ord), 200)          as prompt_preview
from trainer_jobs,
     lateral jsonb_array_elements(job_config->'config'->'process'->0->'sample'->'samples')
       with ordinality as s(pr, ord)
where is_deleted is not true and status = 'stopped'
  and job_config->'config'->'process'->0->'model'->>'arch' = '<ARCH>'
  and (job_config->'config'->'process'->0->'model'->>'quantize') = 'false'
  and coalesce(job_config->'config'->'process'->0->'model'->>'quantize_te','true') = 'false'
  and step >= 2000
group by 1,2,3,4,5,6,7,8,9,10
order by step desc
limit 20;

-- 3) EDIT arches only: you need jobs whose sample prompts carry ctrl_img*.
--    Most edit jobs sample WITHOUT reference images -> they only test the
--    text-to-image path, not reference-latent injection. Also read model_kwargs:
--    kv_cache / match_target_res must be replayed at inference.
select
  id, name, step, user_id,
  job_config->'config'->'process'->0->'model'->'model_kwargs'  as model_kwargs,
  count(*) as n_prompts,
  count(*) filter (where (pr ? 'ctrl_img') or (pr ? 'ctrl_img_1')) as n_with_ctrl
from trainer_jobs,
     lateral jsonb_array_elements(job_config->'config'->'process'->0->'sample'->'samples')
       as s(pr)
where is_deleted is not true and status = 'stopped'
  and job_config->'config'->'process'->0->'model'->>'arch' in ('<ARCH>')   -- e.g. 'krea2:o_edit'
group by 1,2,3,4,5
having count(*) filter (where (pr ? 'ctrl_img') or (pr ? 'ctrl_img_1')) > 0
order by step desc;

-- Public CDN layout (no auth needed for a specific file; directory listing is 404):
--   config.yaml   https://files.runcomfy.net/train/users/<uid>/ai-toolkit/output/<name>/config.yaml
--   LoRA          .../output/<name>/<name>_<step zero-padded to 9>.safetensors
--   control image .../ai-toolkit/data/images/<file>   (URL-encode the filename)
--   samples DIR   .../output/<name>/samples/  <- needs the JuiceFS mount to LIST
--                 (filenames are <ms-timestamp>__<step9>_<idx>.jpg, unpredictable)
