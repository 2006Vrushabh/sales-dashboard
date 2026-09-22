-- =============================================================================
-- Smart Sales Data Mining and Analysis Platform — Database Schema
-- Run this once in the Supabase SQL Editor (SQL Editor -> New Query -> Run)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- TABLE 1 : profiles
-- One row per user. Supabase Auth already has its own hidden "auth.users"
-- table (id, email, encrypted password, etc). We never touch that table
-- directly -- this "profiles" table just holds the extra info we want
-- (the user's display name) and is linked to auth.users by the same id.
-- -----------------------------------------------------------------------------
create table if not exists profiles (
    id         uuid primary key references auth.users(id) on delete cascade,
    name       text not null,
    email      text not null,
    created_at timestamptz not null default now()
);

alter table profiles enable row level security;

-- A user may only see / edit their OWN profile row.
create policy "profiles: select own row"
    on profiles for select
    using (auth.uid() = id);

create policy "profiles: update own row"
    on profiles for update
    using (auth.uid() = id);

-- This function runs automatically every time someone signs up.
-- It copies their name (passed in from the sign-up form) and email into
-- the profiles table, so the app never has to do that step manually.
create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, name, email)
    values (
        new.id,
        coalesce(new.raw_user_meta_data->>'name', 'User'),
        new.email
    );
    return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();


-- -----------------------------------------------------------------------------
-- TABLE 2 : datasets
-- One row per dataset a user has uploaded/analyzed. This combines what the
-- brief called "datasets" and "analyses" into a single table -- each dataset
-- in this app only ever has ONE summary attached to it, so a second table
-- joined by a foreign key would just be extra complexity with no benefit.
-- (This is a deliberate simplification, in line with "avoid unnecessary
-- abstraction" from the project brief.)
-- -----------------------------------------------------------------------------
create table if not exists datasets (
    id                  uuid primary key default gen_random_uuid(),
    user_id             uuid not null references auth.users(id) on delete cascade,
    name                text not null,
    storage_path        text,                    -- where the CSV lives in Storage
    rows                integer not null,
    columns             integer not null,
    numerical_columns   integer not null,
    categorical_columns integer not null,
    missing_values      integer not null,
    duplicate_rows      integer not null,
    -- Which column (if any) was used for each measure. Different CSVs use
    -- different names (Sales vs Revenue), so we remember the actual column
    -- name chosen, and its summary numbers, for later comparison.
    sales_column        text,
    total_sales         double precision,
    average_sales       double precision,
    profit_column        text,
    total_profit         double precision,
    average_profit        double precision,
    quantity_column      text,
    total_quantity        double precision,
    created_at          timestamptz not null default now()
);

alter table datasets enable row level security;

-- A user may only see / insert / update / delete THEIR OWN dataset rows.
create policy "datasets: select own rows"
    on datasets for select
    using (auth.uid() = user_id);

create policy "datasets: insert own rows"
    on datasets for insert
    with check (auth.uid() = user_id);

create policy "datasets: delete own rows"
    on datasets for delete
    using (auth.uid() = user_id);


-- -----------------------------------------------------------------------------
-- STORAGE : bucket policies
-- Run this AFTER creating the "datasets" bucket in the Storage tab.
-- Files are saved as  <user_id>/<filename>.csv  -- these policies make sure
-- a user can only read/write inside their OWN folder (their own user id).
-- -----------------------------------------------------------------------------
create policy "storage: users read own folder"
    on storage.objects for select
    using (bucket_id = 'datasets' and auth.uid()::text = (storage.foldername(name))[1]);

create policy "storage: users upload to own folder"
    on storage.objects for insert
    with check (bucket_id = 'datasets' and auth.uid()::text = (storage.foldername(name))[1]);

create policy "storage: users delete own files"
    on storage.objects for delete
    using (bucket_id = 'datasets' and auth.uid()::text = (storage.foldername(name))[1]);
