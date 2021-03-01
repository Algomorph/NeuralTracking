//  ================================================================
//  Created by nitronoid (https://stackoverflow.com/users/7878485/nitronoid?tab=profile)
//  Copyright (c) 2020 nitronoid
//  Source: https://stackoverflow.com/questions/38955940/how-to-concatenate-static-strings-at-compile-time/62823211#62823211
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once
#ifndef STRING_CONSTANT_H
#define STRING_CONSTANT_H

#include <cstddef>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>


namespace test{

// Recursive comparison of each individual character in a string
// The last bit with std::enable_if uses SFINAE (Substitution Failure Is Not An Error)
// to rule this function out and switch to the base case for the recursion when the Index == Length
template <std::size_t Length, std::size_t Index, typename Left, typename Right>
constexpr auto CompareCharacters( const Left& lhs, const Right& rhs ) -> typename std::enable_if<Index != Length, bool>::type
{
	return lhs[Index] == rhs[Index] ? CompareCharacters<Length, Index + 1>( lhs, rhs ) : false;
}

// Recursion base case. If you run past the last index of
template <std::size_t Length, std::size_t Index, typename Left, typename Right, typename std::enable_if<Index == Length, bool>::type = 0>
constexpr bool CompareCharacters( const Left& lhs, const Right& rhs )
{
	return true;
}


// Helper type traits to determine the length of either
// a string literal or a StringConstant (specialized below)
template <typename T>
struct length_of
{
	static_assert( std::is_void<T>::value, "Unsupported type for length_of" );

	static constexpr std::size_t value = 1;
};

template <std::size_t N>
struct length_of< const char(&)[N] >
{
	static constexpr std::size_t value = N - 1;
};

template <std::size_t N>
struct length_of< char[N] >
{
	static constexpr std::size_t value = N - 1;
};

template <std::size_t N>
struct length_of< const char[N] >
{
	static constexpr std::size_t value = N - 1;
};


// This small class is the heart of the constant string implementation.
// It has constructors for string literals and individual chars, as well
// as operators to interact with string literals or other instances. This
// allows for it to have a very natural interface, and it's all constexpr
// Inspired heavily by a class described in a presentation by Scott Schurr
// at Boostcon:
// https://github.com/boostcon/cppnow_presentations_2012/blob/master/wed/schurr_cpp11_tools_for_class_authors.pdf
template <std::size_t N>
class StringConstant
{
public:
	// Constructor which takes individual chars. Allows for unpacking
	// parameter packs directly into the constructor
	template <typename... Characters>
	constexpr StringConstant( Characters... characters )
			: m_value{ characters..., '\0' }
	{
	}

	// Copy constructor
	template <std::size_t... Indexes>
	constexpr StringConstant( const StringConstant<N>& rhs, std::index_sequence<Indexes...> dummy = StringConstant::g_indexes )
			: m_value{ rhs[Indexes]..., '\0' }
	{
	}

	template <std::size_t X, std::size_t... Indexes>
	constexpr StringConstant( const StringConstant<X>& rhs, std::index_sequence<Indexes...> dummy )
			: m_value{ rhs[Indexes]..., '\0' }
	{
	}

	template <std::size_t... Indexes>
	constexpr StringConstant( const char(&value)[N + 1], std::index_sequence<Indexes...> dummy )
			: StringConstant( value[Indexes]... )
	{
	}

	constexpr StringConstant( const char(&value)[N + 1] )
			: StringConstant( value, std::make_index_sequence<N>{} )
	{
	}

	// Array subscript operator, with some basic range checking
	constexpr char operator[]( const std::size_t index ) const
	{
		return index < N ? m_value[index] : throw std::out_of_range("Index out of range");
	}

	constexpr const char* Get( ) const { return m_value; }
	constexpr std::size_t Length( ) const { return N; }

	std::string ToString( ) const { return std::string( m_value ); }

protected:
	const char m_value[N + 1];

	static constexpr auto g_indexes = typename std::make_index_sequence<N>{};
};

// Specialize the length_of trait for the StringConstant class
template <std::size_t N>
struct length_of< StringConstant<N> >
{
	static constexpr std::size_t value = N;
};

template <std::size_t N>
struct length_of< const StringConstant<N> >
{
	static constexpr std::size_t value = N;
};

template <std::size_t N>
struct length_of< const StringConstant<N>& >
{
	static constexpr std::size_t value = N;
};

// A helper trait for checking if something is a StringConstant
// without having to know the length of the string it contains
template <typename T>
struct is_string_constant
{
	static constexpr bool value = false;
};

template <std::size_t N>
struct is_string_constant< StringConstant<N> >
{
	static constexpr bool value = true;
};

template <std::size_t N>
struct is_string_constant< StringConstant<N>& >
{
	static constexpr bool value = true;
};

template <std::size_t N>
struct is_string_constant< const StringConstant<N> >
{
	static constexpr bool value = true;
};

template <std::size_t N>
struct is_string_constant< const StringConstant<N>& >
{
	static constexpr bool value = true;
};


// A helper function for concatenating StringConstants

// Less than human friendly concat function, wrapped by a huamn friendly one below
template <typename Left, typename Right, std::size_t... IndexesLeft, std::size_t... IndexesRight>
constexpr StringConstant<sizeof...(IndexesLeft) + sizeof...(IndexesRight)> ConcatStrings( const Left& lhs, const Right& rhs, std::index_sequence<IndexesLeft...> dummy1, std::index_sequence<IndexesRight...> dummy2 )
{
	return StringConstant<sizeof...(IndexesLeft) + sizeof...(IndexesRight)>( lhs[IndexesLeft]..., rhs[IndexesRight]... );
}

// Human friendly concat function for string literals
template <typename Left, typename Right>
constexpr StringConstant<length_of<Left>::value + length_of<Right>::value> ConcatStrings( const Left& lhs, const Right& rhs )
{
	return ConcatStrings( lhs, rhs, typename std::make_index_sequence<length_of<decltype(lhs)>::value>{}, typename std::make_index_sequence<length_of<decltype(rhs)>::value>{} );
}


// Finally, operators for dealing with a string literal LHS and StringConstant RHS

// Addition operator
template <std::size_t N, typename Right>
constexpr StringConstant<N + length_of<Right>::value> operator+( const StringConstant<N>& lhs, const Right& rhs )
{
	return ConcatStrings( lhs, rhs );
}

template <typename Left, std::size_t N>
constexpr StringConstant<length_of<Left>::value + N> operator+( const Left& lhs, const StringConstant<N>& rhs )
{
	return ConcatStrings( lhs, rhs );
}

template <std::size_t X, std::size_t Y>
constexpr StringConstant<X + Y> operator+( const StringConstant<X>& lhs, const StringConstant<Y>& rhs )
{
	return ConcatStrings( lhs, rhs );
}

// Equality operator
template <std::size_t N, typename Right>
constexpr auto operator==( const StringConstant<N>& lhs, const Right& rhs ) -> typename std::enable_if<N == length_of<Right>::value, bool>::type
{
	return CompareCharacters<N, 0>( lhs, rhs );
}

template <typename Left, std::size_t N>
constexpr auto operator==( const Left& lhs, const StringConstant<N>& rhs ) -> typename std::enable_if<length_of<Left>::value == N, bool>::type
{
	return CompareCharacters<N, 0>( lhs, rhs );
}

template <std::size_t X, std::size_t Y>
constexpr auto operator==( const StringConstant<X>& lhs, const StringConstant<Y>& rhs ) -> typename std::enable_if<X == Y, bool>::type
{
	return CompareCharacters<X, 0>( lhs, rhs );
}

// Different length strings can never be equal
template <std::size_t N, typename Right, typename std::enable_if<N != length_of<Right>::value, bool>::type = 0>
constexpr bool operator==( const StringConstant<N>& lhs, const Right& rhs )
{
	return false;
}

// Different length strings can never be equal
template <typename Left, std::size_t N, typename std::enable_if<length_of<Left>::value != N, bool>::type = 0>
constexpr bool operator==( const Left& lhs, const StringConstant<N>& rhs )
{
	return false;
}

// Different length strings can never be equal
template <std::size_t X, std::size_t Y, typename std::enable_if<X != Y, bool>::type = 0>
constexpr bool operator==( const StringConstant<X>& lhs, const StringConstant<Y>& rhs )
{
	return false;
}


template <std::size_t N, std::size_t... Indexes>
constexpr auto StringFactory( const char(&value)[N], std::index_sequence<Indexes...> dummy )
{
	return StringConstant<N - 1>( value[Indexes]... );
}


// A helper factory function for creating FixedStringConstant objects
// which handles figuring out the length of the string for you
template <std::size_t N>
constexpr auto StringFactory( const char(&value)[N] )
{
	return StringFactory( value, typename std::make_index_sequence<N - 1>{} );
}


} // namespace test
#endif