#pragma once

#include <algorithm>
#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

// Allocation of aligned memory in pre - C++17
#ifdef _MSC_VER
#include <malloc.h>
#elif !(defined ISOC11_SOURCE) && (__cplusplus < 201703L)
#include <xmmintrin.h>
#endif

#include <cstdlib>
#include <cmath>
#include <cassert>

namespace mw {

/**
 * An unordered multi-map based on Robin-Hood- and Fibonacci-Hashing.
 *
 * @author	Michael Weitzel
 */
template<
	typename Key,
	typename Value,
	typename KeyHash = std::hash<Key>,
	typename KeyCompare = std::equal_to<Key>
	>
class hash_multimap
{
private:
	/* the state (EMPTY|DELETED) is encoded in the two most significant bits */

	/** an empty slot */
	static constexpr std::size_t EMPTY = // 0x40000...
		(1ul<<(8*sizeof(std::size_t)-2));
	/** a deleted slot */
	static constexpr std::size_t DELETED = // 0x80000...
		(2ul<<(8*sizeof(std::size_t)-2));
	/** a constant used for "not found" */
	static constexpr std::size_t NIL = std::size_t(-1);
	/** a mask to exclude the two most significant bits (state) */
	static constexpr std::size_t HASH_MASK = // 0x3ffff...
		~(3ul<<(8*sizeof(std::size_t)-2));

	template< bool is_const > class iterator_template;

public:
	using key_type = Key;
	using mapped_type = Value;
	using value_type = std::pair<const key_type,mapped_type>;
	using size_type = std::size_t;
	using hasher = KeyHash;
	using key_equal = KeyCompare;
	//using allocator_type = ...
	using reference = value_type&;
	using const_reference = const value_type&;
	using pointer = value_type*;
	using const_pointer = const value_type*;
	using iterator = iterator_template<false>;
	using const_iterator = iterator_template<true>;

public:
	hash_multimap() = default;

	inline hash_multimap(hash_multimap&& cpy)
		: hash_values_(cpy.hash_values_)
		, table_(cpy.table_)
		, capacity_(cpy.capacity_)
		, grow_threshold_(cpy.grow_threshold_)
		, size_(cpy.size_)
		, mask_(cpy.mask_)
		, ld_capacity_(cpy.ld_capacity_)
		, max_load_factor_(cpy.max_load_factor_)
		, hash_(std::move(cpy.hash_))
		, key_eq_(std::move(cpy.key_eq_))
	{
		cpy.hash_values_ = nullptr;
		cpy.table_ = nullptr;
		cpy.capacity_ = 0;
	}
	hash_multimap& operator=(hash_multimap&& cpy)
	{
		if (&cpy == this)
			return *this;

		clear();
		if (table_ != nullptr)
			deallocate();
		hash_values_ = cpy.hash_values_;
		table_ = cpy.table_;
		capacity_ = cpy.capacity_;
		grow_threshold_ = cpy.grow_threshold_;
		size_ = cpy.size_;
		mask_ = cpy.mask_;
		ld_capacity_ = cpy.ld_capacity_;
		max_load_factor_ = cpy.max_load_factor_;
		hash_ = std::move(cpy.hash_);
		key_eq_ = std::move(cpy.key_eq_);
		cpy.hash_values_ = nullptr;
		cpy.table_ = nullptr;
		cpy.capacity_ = 0;
		return *this;
	}

	hash_multimap(const hash_multimap& cpy)
	{
		if (&cpy == this)
			return;

		clear();
		if (capacity_ != cpy.capacity_)
		{
			if (table_ != nullptr)
				deallocate();
			capacity_ = cpy.capacity_;
			ld_capacity_ = cpy.ld_capacity_;
			if (capacity_ == 0)
			{
				table_ = nullptr;
				hash_values_ = nullptr;
			}
			else
				allocate();
		}

		for (size_type i=0; i<cpy.capacity_; ++i)
		{
			const value_type& v = cpy.table_[i];
			size_type h = cpy.hash_values_[i];

			if (is_occupied(h))
				insert(h,value_type(v));
		}
		grow_threshold_ = cpy.grow_threshold_;
		size_ = cpy.size_;
		mask_ = cpy.mask_;
		max_load_factor_ = cpy.max_load_factor_;
		hash_ = cpy.hash_;
		key_eq_ = cpy.key_eq_;
	}
	inline hash_multimap& operator=(const hash_multimap& cpy)
	{
		return operator=(hash_multimap(cpy));
	}

	inline hash_multimap& operator=(
		const std::initializer_list<value_type>& values
		)
	{
		clear();
		insert(values);
		return *this;
	}

	~hash_multimap()
	{
		if (table_ != nullptr)
		{
			clear();
			deallocate();
		}
	}

	inline explicit hash_multimap(
		size_type reserved_size,
		const hasher& hash = hasher(),
		const key_equal& key_eq = key_equal()
		)
		: hash_(hash)
		, key_eq_(key_eq)
	{
		reserve(reserved_size);
	}

	hash_multimap(
		const std::initializer_list< value_type >& values,
		const hasher& hash = hasher(),
		const key_equal& key_eq = key_equal()
		)
		: hash_multimap(values.size(),hash,key_eq)
	{
		insert(values);
	}

	void clear()
	{
		for (size_type slot=0; slot<capacity_; ++slot)
		{
			if (is_occupied(hash_values_[slot]))
				table_[slot].~value_type();
			hash_values_[slot] = EMPTY;
		}
		size_ = 0;
	}

	inline const mapped_type& at(const key_type& key) const
	{
		size_type slot = lookup_slot(key);
		if (slot != NIL)
			return table_[slot].second;
		throw std::out_of_range("invalid key");
	}

	inline mapped_type& at(const key_type& key)
	{
		size_type slot = lookup_slot(key);
		if (slot != NIL)
			return table_[slot].second;
		throw std::out_of_range("invalid key");
	}

	inline mapped_type& operator[](const key_type& key)
	{
		size_type slot, h = hash_key(key);
		if (empty() || (slot = lookup_slot(h,key)) == NIL)
		{
			if (++size_ >= grow_threshold_)
				grow();
			slot = insert(h,{key,mapped_type()});
		}
		return table_[slot].second;
	}

	inline mapped_type& operator[](key_type&& key)
	{
		size_type slot, h = hash_key(key);
		if (empty() || (slot = lookup_slot(h,key)) == NIL)
		{
			if (++size_ >= grow_threshold_)
				grow();
			slot = insert(h,{std::forward<key_type>(key),mapped_type()});
		}
		return table_[slot].second;
	}

	inline const_iterator find(const key_type& key) const noexcept
	{
		size_type slot = lookup_slot(key);
		return slot == NIL
			? const_iterator()
			: const_iterator(this,slot);
	}

	inline iterator find(const key_type& key) noexcept
	{
		size_type slot = lookup_slot(key);
		return slot == NIL
			? iterator()
			: iterator(this,slot);
	}

	std::pair<iterator,bool> insert(value_type&& value)
	{
		if (++size_ >= grow_threshold_)
			grow();
		size_type h = hash_key(value.first);
		size_type slot = insert(h,std::forward<value_type>(value));
		return { iterator(this,slot),true };
	}

	inline std::pair<iterator,bool> insert(const value_type& value)
	{
		return insert(value_type(value));
	}

	void insert(
		const std::initializer_list< value_type >& values
		)
	{
		for (const value_type& v : values)
			insert(v);
	}

	template< typename InputIt >
	inline void insert(
		InputIt first,
		const InputIt& last
		)
	{
		for (; first != last; ++first)
			insert(*first);
	}

	template< typename... Args >
	inline std::pair<iterator,bool> emplace(Args&&... args)
	{
		return insert(value_type(std::forward<Args>(args)...));
	}

	inline bool erase_one(const key_type& key)
	{
		size_type slot = lookup_slot(key);
		if (slot == NIL)
			return false;

		table_[slot].~value_type();
		hash_values_[slot] |= DELETED;
		size_--;
		return true;
	}

	size_type erase(const key_type& key)
	{
		size_type slot = lookup_slot(key);
		if (slot == NIL)
			return 0ul;
		size_type h = hash_values_[slot];
		size_type nerased = 0ul;
		iterator iter(this,slot);
		do
		{
			table_[iter.slot_].~value_type();
			hash_values_[iter.slot_] |= DELETED;
			++nerased;
			++iter;
		}
		while (iter.slot_ < capacity_
			&& hash_values_[iter.slot_] == h
			&& key_eq_(iter->first,key)
			);
		size_ -= nerased;
		return nerased;
	}

	inline iterator erase(const_iterator iter)
	{
		if (!iter)
			return iter;

		assert(is_occupied(hash_values_[iter.slot_]));

		table_[iter.slot_].~value_type();
		hash_values_[iter.slot_] |= DELETED;
		size_--;
		return ++iter;
	}

	inline iterator erase(
		const_iterator first,
		const_iterator last
		)
	{
		while (bool(first) && first != last)
			first = erase(first);
		return first;
	}

	inline value_type extract(const key_type& key) { return extract(find(key)); }

	inline value_type extract(const_iterator iter)
	{
		if (!bool(iter))
			return value_type();
		value_type ex(std::move(table_[iter.slot_]));
		table_[iter.slot_].~value_type();
		hash_values_[iter.slot_] |= DELETED;
		size_--;
		return ex;
	}

	void shrink_to_fit()
	{
		size_type reserved_size =
			size_type(std::ceil(size_ / max_load_factor_));
		unsigned int ceil_ld_size = ceil_ld(reserved_size);

		if (ceil_ld_size < ld_capacity_)
			rehash(std::max(ceil_ld_size,3u));
	}

	void reserve(size_type size)
	{
		size_type reserved_size =
			size_type(std::ceil(size / max_load_factor_));
		unsigned int ceil_ld_size = ceil_ld(reserved_size);

		if (ceil_ld_size > ld_capacity_) // no shrinking!
			rehash(std::max(ceil_ld_size,3u));
	}

	inline size_type capacity() const noexcept { return capacity_; }
	inline size_type max_size() const noexcept { return HASH_MASK; }
	inline size_type size() const noexcept { return size_; }
	inline float load_factor() const noexcept { return float(size_)/capacity_; }
	inline float max_load_factor() const noexcept { return max_load_factor_; }
	inline void max_load_factor(float mlf)
	{
		if (mlf <= 0.f || mlf >= 1.f)
			throw std::invalid_argument("invalid load factor");
		max_load_factor_ = mlf;
	}
	inline bool empty() const noexcept { return size() == 0; }
	inline bool contains(const key_type& key) const noexcept
	{
		return lookup_slot(key) != NIL;
	}

	size_type count(const key_type& key) const noexcept
	{
		size_type slot = lookup_slot(key);
		if (slot == NIL)
			return 0ul;
		size_type h = hash_values_[slot];
		size_type n = 0ul;
		iterator iter(this,slot);
		do
		{
			++n;
			++iter;
		}
		while (iter.slot_ < capacity_
			&& hash_values_[iter.slot_] == h
			&& key_eq_(iter->first,key)
			);
		return n;
	}

	inline void swap(hash_multimap& other)
	{
		std::swap(hash_,other.hash_);
		std::swap(key_eq_,other.key_eq_);
		std::swap(hash_values_,other.hash_values_);
		std::swap(table_,other.table_);
		std::swap(ld_capacity_,other.ld_capacity_);
		std::swap(grow_threshold_,other.grow_threshold_);
		std::swap(size_,other.size_);
		std::swap(mask_,other.mask_);
		std::swap(max_load_factor_,other.max_load_factor_);
	}

	/* Observers */

	inline hasher hash_function() const { return hash_; }
	inline key_equal key_eq() const { return key_eq_; }

private:
	inline static unsigned int ceil_ld(size_type v) noexcept
	{
		unsigned int ceil_ld_v = 0u;
		for (size_type n=v; n>1ul; n>>=1u)
			ceil_ld_v++;
		if (v > (1ul<<ceil_ld_v))
			ceil_ld_v++;
		return ceil_ld_v;
	}

	void allocate()
	{
#if (defined _ISOC11_SOURCE) || (__cplusplus >= 201703L)
		using namespace std;
		table_ = reinterpret_cast<value_type*>(aligned_alloc(
					std::alignment_of<value_type>::value, // C++11
					capacity_*sizeof(value_type)));
#elif defined _MSC_VER
		table_ = reinterpret_cast<value_type*>(_aligned_malloc(
					capacity_*sizeof(value_type),
					std::alignment_of<value_type>::value)); // C++11
#else
		table_ = reinterpret_cast<value_type*>(_mm_malloc(
					capacity_*sizeof(value_type),
					std::alignment_of<value_type>::value)); // C++11
#endif
		hash_values_ = new size_type[capacity_];
		for (size_type i=0; i<capacity_; ++i)
			hash_values_[i] = EMPTY;

		grow_threshold_ = size_type(max_load_factor_ * capacity_ + .5f);
		mask_ = capacity_ - 1ul;
	}

	void deallocate()
	{
#if (defined ISOC11_SOURCE) || (__cplusplus >= 201703L)
		using namespace std;
		free(table_);
#elif defined _MSC_VER
		_aligned_free(table_);
#else
		_mm_free(table_);
#endif
		delete[] hash_values_;
	}

	inline void grow()
	{
		rehash(capacity_ == 0 ? 3u : ld_capacity_ + 1u);
	}

	void rehash(unsigned int ld_capacity)
	{
		value_type* old_table = table_;
		size_type old_capacity = capacity_;
		size_type* old_hash_values = hash_values_;

		ld_capacity_ = ld_capacity;
		capacity_ = 1ul<<ld_capacity_;

		allocate();

		// alte Werte übernehmen
		for (size_type i=0; i<old_capacity; ++i)
		{
			value_type& v = old_table[i];
			size_type h = old_hash_values[i];

			if (is_occupied(h))
			{
				insert(h,std::move(v));
				v.~value_type();
			}
		}

		if (old_table)
		{
#if (defined ISOC11_SOURCE) || (__cplusplus >= 201703L)
			using namespace std;
			free(old_table);
#elif defined _MSC_VER
			_aligned_free(old_table);
#else
			_mm_free(old_table);
#endif
			delete[] old_hash_values;
		}
	}

	inline size_type hash_key(const key_type& key) const
	{
		// Golden Ratio: (a+b)/a = a/b = (1+sqrt(5))/2 = phi
		constexpr long double phi =
			1.6180339887498948482045868343656381177203091798057628621354486227L;
		// division with truncation
		constexpr size_type fib_wrap = static_cast<size_type>(
			std::numeric_limits<size_type>::max()/phi);
		// closest odd number (11400714819323198485)
		constexpr size_type fib_mul = (fib_wrap & 1ul) ? fib_wrap : fib_wrap+1ul;

		static_assert(sizeof(size_type)!=8 || fib_mul == 11400714819323198485ul,
			"computer is wrong");

		return (hash_(key) * fib_mul) >> 2;
	}

	inline size_type slot_index(
		size_type hash_value
		) const
	{
		constexpr unsigned int shift = 8u*sizeof(size_type) - 2u;
		return hash_value >> (shift - ld_capacity_);
	}

	inline static bool is_deleted(size_type hash_value)
	{
		return (hash_value & DELETED) != 0ul;
	}

	inline static bool is_empty(size_type hash_value)
	{
		return (hash_value & EMPTY) != 0ul;
	}

	inline static bool is_occupied(size_type hash_value)
	{
		return (hash_value & HASH_MASK) == hash_value;
	}

	inline size_type probe_distance(
		size_type slot
		) const
	{
		return (slot + capacity_ - slot_index(hash_values_[slot])) & mask_;
	}

	inline void emplace(
		size_type hash_value,
		size_type slot,
		value_type && value
		)
	{
		new (&table_[slot]) value_type(std::forward<value_type>(value));
		hash_values_[slot] = hash_value;
	}

	size_type insert(
		size_type hash_value,
		value_type && value
		)
	{
		size_type slot = slot_index(hash_value);
		for (size_type dist=0;;++dist)
		{
			if (is_empty(hash_values_[slot]))
			{
				emplace(hash_value,slot,std::forward<value_type>(value));
				return slot;
			}

			// If the existing elem has probed less than us, then swap
			// places with existing elem, and keep going to find another
			// slot for that elem.
			size_type existing_elem_probe_dist =
				probe_distance(slot);
			if (existing_elem_probe_dist < dist)
			{
				if (is_deleted(hash_values_[slot]))
				{
					emplace(hash_value,slot,std::forward<value_type>(value));
					return slot;
				}

				std::swap(hash_value, hash_values_[slot]);
				std::swap(const_cast<key_type&>(value.first),
					const_cast<key_type&>(table_[slot].first));
				std::swap(value.second,table_[slot].second);

				dist = existing_elem_probe_dist;
			}

			slot = (slot + 1ul) & mask_;
		}
	}

	inline size_type lookup_slot(const key_type& key) const noexcept
	{
		if (empty())
			return NIL;
		return lookup_slot(hash_key(key),key);
	}

	size_type lookup_slot(size_type h, const key_type& key) const noexcept
	{
		size_type slot = slot_index(h);
		for (size_type dist=0;;++dist)
		{
			if (is_empty(hash_values_[slot]) || dist > probe_distance(slot))
				return NIL;
			else if (hash_values_[slot] == h && key_eq_(table_[slot].first,key))
				return slot;
			slot = (slot + 1ul) & mask_;
		}
	}

private:
	template< bool is_const > class iterator_template
	{
		friend class hash_multimap;
		friend class iterator_template<false>;
	public:
		using iterator_category = std::forward_iterator_tag;
		using hash_multimap_type = typename std::conditional< is_const,
			const hash_multimap<Key,Value,KeyHash,KeyCompare>,
			hash_multimap<Key,Value,KeyHash,KeyCompare>
			>::type;
		using value_type = typename hash_multimap_type::value_type;
		using difference_type = size_type;
		using pointer = value_type*;
		using reference = value_type&;

	private:
		const hash_multimap * obj_ = nullptr;
		size_type slot_ = 0ul;

		inline iterator_template(
			const hash_multimap * obj,
			size_type slot
			) : obj_(obj), slot_(slot)
		{
			// the given slot must not be empty
			assert(slot_==obj->capacity_ || is_occupied(obj_->hash_values_[slot_]));
		}

		/** Conversion iterator_template<true> -> iterator_template<false> */
		template< typename T > inline iterator_template(
			const T& cpy,
			typename std::enable_if<
				!is_const && std::is_same<T,iterator_template<true>>::value
				>::type * = nullptr
			) : obj_(cpy.obj_), slot_(cpy.slot_) { }

	public:
		iterator_template() = default;
		iterator_template(const iterator_template&) = default;
		iterator_template& operator=(const iterator_template&) = default;

		/** Conversion iterator_template<false> -> iterator_template<true> */
		template< typename T > inline iterator_template(
			const T& cpy,
			typename std::enable_if<
				is_const && std::is_same<T,iterator_template<false>>::value
				>::type * = nullptr
			) : obj_(cpy.obj_), slot_(cpy.slot_) { }

		inline iterator_template& operator++() noexcept
		{
			size_type capacity = obj_->capacity();
			for (;;)
			{
				++slot_;
				if (slot_ >= capacity) return *this;
				if (is_occupied(obj_->hash_values_[slot_]))
					return *this;
			}
		}
		inline iterator_template& operator++(int) noexcept
		{
			iterator_template prev(*this);
			operator++();
			return prev;
		}
		inline reference operator*() const noexcept { return obj_->table_[slot_]; }
		inline pointer operator->() const noexcept { return &obj_->table_[slot_]; }
		inline bool operator==(const iterator_template& r) const noexcept
		{
			return slot_ == r.slot_;
		}
		inline bool operator!=(const iterator_template& r) const noexcept
		{
			return slot_ != r.slot_;
		}
		inline explicit operator bool() const noexcept
		{
			return slot_<obj_->capacity() && is_occupied(obj_->hash_values_[slot_]);
		}

	};

public:
	inline iterator begin() noexcept
	{
		// advance to the first occupied slot
		size_type slot;
		for (slot=0; slot<capacity_; ++slot)
			if (is_occupied(hash_values_[slot]))
				break;
		return iterator(this,slot);
	}
	inline iterator end() noexcept
	{
		return iterator(this,capacity_);
	}

	inline const_iterator begin() const noexcept
	{
		return const_iterator(const_cast<hash_multimap>(*this).begin());
	}
	inline const_iterator end() const noexcept
	{
		return const_iterator(this,capacity_);
	}

	inline const_iterator cbegin() const noexcept { return begin(); }
	inline const_iterator cend() const noexcept { return end(); }

	std::pair<iterator,iterator> equal_range(
		const key_type& key
		) noexcept
	{
		size_type slot = lookup_slot(key);
		if (slot == NIL)
			return {end(),end()};
		size_type h = hash_values_[slot];
		iterator start(this,slot);
		iterator stop(start); ++stop;
		while (stop.slot_ < capacity_
			&& hash_values_[stop.slot_] == h
			&& key_eq_(stop->first,key)
			) ++stop;
		return {start,stop};
	}

	inline std::pair<const_iterator,const_iterator> equal_range(
		const key_type& key
		) const noexcept
	{
		std::pair<iterator,iterator> be =
			const_cast<hash_multimap>(*this).equal_range(key);
		return {be.first,be.second};
	}

	float average_probe_count() const noexcept
	{
		float probe_total = 0.f;
		for(size_type i=0; i<capacity_; ++i)
		{
			size_type h = hash_values_[i];
			if (is_occupied(h))
				probe_total += (float)probe_distance(i);
		}
		return probe_total / (float)size() + 1.f;
	}

private:
	size_type *hash_values_ = nullptr;
	value_type *table_ = nullptr;
	size_type capacity_ = 0;
	size_type grow_threshold_ = 0;
	size_type size_ = 0;
	size_type mask_ = 0;
	unsigned int ld_capacity_ = 0;
	float max_load_factor_ = 0.9f;
	hasher hash_;
	key_equal key_eq_;

};

} // namespace mw

// vim:set fenc=utf-8:
